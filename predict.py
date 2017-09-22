import torch
from torch.autograd import Variable
import pickle
from Datahelper2 import *
from Model import *
from score_thresholds import *
import numpy
import os,sys
from gloable_parameter import *


torch.backends.cudnn.benchmark=True


def predict(model_path,batch_size,resize,gpu):

    folds = 5
    transform_num = 8

    globals_res  = torch.zeros((test_num,17))
    names = []

    predict_np = [[],[],[],[],[]] # predict_np[k][i] : predict on fold[k] type[i]
    for fold in range(folds):
        print('Predicting fold%d.mod'%fold)
        model_name = os.path.join(model_path,'fold%d.mod'%fold)
        model = torch.load(model_name)
        model.eval()
        model.cuda(gpu)

        total_res = torch.zeros((test_num, 17))

        for type in range(transform_num):
            print('predicting with tpye ' + str(type) + '...')
            res_list = []
            dset_test = KaggleAmazonDataset_test(IMG_TEST_PATH,transform_type=type,resize=resize)
            test_loader = DataLoader(dset_test, batch_size=batch_size, num_workers=6)
            for step, (data, name) in enumerate(test_loader):
                if type==0 and fold==0:
                    names.extend(name)
                data = Variable(data,volatile=True).cuda(gpu)
                output = model(data)
                res_list.append(output.cpu())
                if (step + 1) % 100 == 0:
                    print('{} Fold{} Type {}: {}/61191 ({:.0f}%)'.format(model_path.split('/')[-1],fold,type, batch_size * (step + 1),
                                                               100. * batch_size * (step + 1) / 61191))
            res = torch.cat(res_list).data
            predict_np[fold].append(res.numpy())
            total_res += res
        mean_res = total_res / transform_num  #tensor
        globals_res += mean_res


    globals_res = (globals_res/folds).numpy()
    with open(os.path.join(model_path,'mean_predict_np.pkl'),'wb') as f:
        pickle.dump(globals_res,f)
    with open(os.path.join(model_path,'name_list.pkl'),'wb') as f:
        pickle.dump(names,f)
    with open(os.path.join(model_path,'predict_np.pkl'),'wb') as f:
        pickle.dump(predict_np,f)

    classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
               'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']

    with open(os.path.join(model_path,'best_th.pkl'),'rb') as f:
        threshold = pickle.load(f)

    print('Writting File...')
    with open(os.path.join(model_path,model_path.split('/')[-1]+'_result'),'w') as f:
        f.write('image_name,tags\n')
        for id in range(globals_res.shape[0]):
            s = ''
            for i,v in enumerate(globals_res[id]):
                if v>=threshold[i]:
                    s += classes[i]+' '
            s = names[id]+','+s+'\n'
            f.write(s)


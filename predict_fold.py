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


def predict_fold(model_path,fold,batch_size,resize,gpu):

    transform_num = 8

    globals_res  = torch.zeros((test_num,17))
    names = []

    print('Predicting fold%d.mod' % fold)
    model_name = os.path.join(model_path, 'fold%d.mod' % fold)
    model = torch.load(model_name)
    model.eval()
    model.cuda(gpu)

    predict_fold_np = []

    for type in range(transform_num):
        print('predicting with tpye ' + str(type) + '...')
        res_list = []
        dset_test = KaggleAmazonDataset_test(IMG_TEST_PATH, transform_type=type, resize=resize)
        test_loader = DataLoader(dset_test, batch_size=batch_size, num_workers=6)
        for step, (data, name) in enumerate(test_loader):
            if type == 0 and fold == 0:
                names.extend(name)
            data = Variable(data, volatile=True).cuda(gpu)
            output = model(data)
            res_list.append(output.cpu())
            if (step + 1) % 100 == 0:
                print('{} Fold{} Type {}: {}/61191 ({:.0f}%)'.format(model_path.split('/')[-1], fold, type,
                                                                     batch_size * (step + 1),
                                                                     100. * batch_size * (step + 1) / 61191))
        res = torch.cat(res_list).data
        predict_fold_np.append(res.numpy())

    with open(os.path.join('predict_fold%d_np.pkl'%fold),'rb') as f:
        pickle.dump(predict_fold_np,f)

predict_fold('../amazon2/vgg11',0,128,224,3)
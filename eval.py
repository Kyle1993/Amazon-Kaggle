import torch
from torch.autograd import Variable
import argparse
import pickle
from sklearn.metrics import fbeta_score
import os
from Datahelper2 import *
from Model import *
from score_thresholds import *
from gloable_parameter import *

torch.backends.cudnn.benchmark=True


def eval(model_path,batch_size,resize,validate_gpu):

    # validate_gpu = 0
    # batch_size = 128
    # model_path = '../amazon2/alexnet'

    folds = 5
    transform_num = 8

    with open('kdf.pkl', 'rb') as f:
        kfold = pickle.load(f,encoding='latin1')

    loss_info = []
    eval_np = [[],[],[],[],[]] #eval_np[k][i] : output on fold[k] type[i]
    labels_np = []   # labels  (train_len,17) : [fold0,fold1...fold4]
    globals_labels_list = []   # merge all the labels  (8*train_len,17)
    globals_output_list = []    # merge all the output (8*train_len,17)
    for fold in range(folds):
        print(model_path.split('/')[-1]+' Validating fold%d.mod'%fold)
        validate_index = kfold[fold][1]
        model_name = os.path.join(model_path,'fold%d.mod'%fold)
        model = torch.load(model_name)
        model.eval()
        model.cuda(validate_gpu)

        # total_res = torch.zeros((validate_num, 17))
        fold_output_list = []
        fold_labels_list = []

        for type in range(transform_num):
            print('validating with tpye ' + str(type) + '...')
            fold_type_output_list = []
            dset_validate = AmazonDateset_validate(validate_index, IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH, transform_type=type,resize=resize)
            validate_loader = DataLoader(dset_validate, batch_size=batch_size, num_workers=6)
            for step, (data, target) in enumerate(validate_loader):
                if type==0:
                    labels_np.append(target)
                data = Variable(data,volatile=True).cuda(validate_gpu)
                output = model(data)

                fold_type_output_list.append(output.data.cpu())
                fold_output_list.append(output.data.cpu())
                fold_labels_list.append(target)
                globals_labels_list.append(target)
                globals_output_list.append(output.data.cpu())


            fold_type_output = torch.cat(fold_type_output_list)
            eval_np[fold].append(fold_type_output.numpy())
        fold_output = torch.cat(fold_output_list)
        fold_labels = torch.cat(fold_labels_list)
        loss = F.binary_cross_entropy(Variable(fold_output,volatile=True),Variable(fold_labels,volatile=True))
        loss_info.append(loss.data[0])



    globals_output = torch.cat(globals_output_list)
    globals_labels = torch.cat(globals_labels_list)
    globals_loss = F.binary_cross_entropy(Variable(globals_output,volatile=True),Variable(globals_labels,volatile=True))
    print('optimize thresholds...')
    best_th = optimise_f2_thresholds(globals_labels.numpy(), globals_output.numpy(), verbose=True)
    score = getScore(globals_labels.numpy(), globals_output.numpy(), best_th)
    # validate_info.append((loss.data[0],best_th,score))
    print('---------------------------------')
    print(model_path.split('/')[-1]+':')
    print('Validate Loss:%f\tValidate Score: %f'%(globals_loss.data[0],score))
    print('Kfolds loss:')
    for l in loss_info:
        print(l)

    labels_np = torch.cat(labels_np).numpy()

    with open(os.path.join(model_path,'validation_globals_output_np.pkl'),'wb') as f:
        pickle.dump(globals_output.numpy(),f)
    with open(os.path.join(model_path,'validation_globals_labels_np.pkl'),'wb') as f:
        pickle.dump(globals_labels.numpy(),f)
    with open(os.path.join(model_path,'validation_labels_np.pkl'),'wb') as f:
        pickle.dump(labels_np,f)
    with open(os.path.join(model_path,'validation_middleoutput_np.pkl'),'wb') as f:
        pickle.dump(eval_np,f)

    with open(os.path.join(model_path,'best_th.pkl'),'wb') as f:
        pickle.dump(best_th,f)
    with open(os.path.join(model_path,'loss_info.pkl'),'wb') as f:
        pickle.dump(loss_info,f)
    with open(os.path.join(model_path,'eval_record'),'w') as f:
        f.write('------------------------------------\n')
        f.write('Validate Loss:%f\tValidate Score: %f\n'%(globals_loss.data[0],score))
        for l in loss_info:
            f.write(str(l)+'\n')

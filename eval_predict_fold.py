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
            if type == 0:
                names.extend(name)
            data = Variable(data, volatile=True).cuda(gpu)
            output = model(data)
            res_list.append(output.cpu())
            if (step + 1) % 10 == 0:
                print('{} Fold{} Type {}: {}/61191 ({:.0f}%)'.format(model_path.split('/')[-1], fold, type,
                                                                     batch_size * (step + 1),
                                                                     100. * batch_size * (step + 1) / 61191))
        res = torch.cat(res_list).data
        predict_fold_np.append(res.numpy())

    with open(os.path.join(model_path,'predict_fold%d_np.pkl'%fold),'wb') as f:
        pickle.dump(predict_fold_np,f)
    with open(os.path.join(model_path,'name_list.pkl'),'wb') as f:
        pickle.dump(names,f)

def eval_fold(model_path,fold,batch_size,resize,validate_gpu):

    transform_num = 8

    with open('kdf.pkl', 'rb') as f:
        kfold = pickle.load(f,encoding='latin1')

    eval_np = [[],[],[],[],[]] #eval_np[k][i] : output on fold[k] type[i]
    labels_np = []   # labels  (train_len,17) : [fold0,fold1...fold4]
    globals_labels_list = []   # merge all the labels  (8*train_len,17)
    globals_output_list = []    # merge all the output (8*train_len,17)

    print(model_path.split('/')[-1]+' Validating fold%d.mod'%fold)
    validate_index = kfold[fold][1]
    model_name = os.path.join(model_path,'fold%d.mod'%fold)
    model = torch.load(model_name)
    model.eval()
    model.cuda(validate_gpu)

    # total_res = torch.zeros((validate_num, 17))
    fold_output_list = []
    fold_labels_list = []

    eval_fold_np = []

    for type in range(transform_num):
        print('validating with tpye ' + str(type) + '...')
        fold_type_output_list = []
        dset_validate = AmazonDateset_validate(validate_index, IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH, transform_type=type,resize=resize)
        validate_loader = DataLoader(dset_validate, batch_size=batch_size, num_workers=6)
        for step, (data, target) in enumerate(validate_loader):

            data = Variable(data,volatile=True).cuda(validate_gpu)
            output = model(data)

            fold_type_output_list.append(output.cpu())
            fold_output_list.append(output.cpu())
            fold_labels_list.append(target)
            # globals_labels_list.append(target)
            # globals_output_list.append(output.data.cpu())


        fold_type_output = torch.cat(fold_type_output_list).data
        eval_fold_np.append(fold_type_output.numpy())
        # eval_np[fold].append(fold_type_output.numpy())
    # fold_output = torch.cat(fold_output_list)
    # fold_labels = torch.cat(fold_labels_list)
    # labels = torch.cat(labels_np)
    fold_output = torch.cat(fold_output_list)
    fold_labels = Variable(torch.cat(fold_labels_list))
    loss = F.binary_cross_entropy(fold_output,fold_labels)
    print(loss.data[0])

    with open(os.path.join(model_path,'eval_fold%d_np.pkl'%fold),'wb') as f:
        pickle.dump(eval_fold_np,f)

def merge_folds(model_path):

    folds = 5
    eval_np = [[],[],[],[],[]] #eval_np[k][i] : output on fold[k] type[i]
    predict_np = [[],[],[],[],[]] # predict_np[k][i] : predict on fold[k] type[i]

    for fold in range(folds):
        with open(os.path.join(model_path,'predict_fold%d_np.pkl'%fold),'rb') as f:
            predict_fold_np = pickle.load(f)
        with open(os.path.join(model_path,'eval_fold%d_np.pkl'%fold),'rb') as f:
            eval_fold_np = pickle.load(f)

        predict_np[fold] = predict_fold_np
        eval_np[fold] = eval_fold_np

    with open(os.path.join(model_path,'validation_middleoutput_np.pkl'),'wb') as f:
        pickle.dump(eval_np,f)
    print('Validation Done!')
    with open(os.path.join(model_path,'predict_np.pkl'),'wb') as f:
        pickle.dump(predict_np,f)
    print('Predict Done!')

if __name__=='__main__':

    model_path = '../amazon2/vgg11'
    gpu = 3
    fold = 4
    batch_size = 128

    # predict_fold(model_path,fold,batch_size,224,gpu)
    # eval_fold(model_path,fold,batch_size,224,gpu)
    merge_folds(model_path)
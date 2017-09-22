import torch
from torch.autograd import Variable
import argparse
import copy
import pickle
from Datahelper2 import *
from Model import *
from gloable_parameter import *
import numpy as np

resnet152 = '../amazon2/resnet152'
resnet50 = '../amazon2/resnet50'
densenet161 = '../amazon2/densenet161'
resnet18 = '../amazon2/resnet18'
densenet161_new = '../amazon2/densenet161_new'
densenet201_new = '../amazon2/densenet201_new'
densenet169_new = '../amazon2/densenet169_new'
resnet34_new = '../amazon2/resnet34_new'
resnet50_new = '../amazon2/resnet50_new'
resnet18_new = '../amazon2/resnet18_new'
resnet101_new = '../amazon2/resnet101_new'
vgg11 = '../amazon2/vgg11'
alexnet = '../amazon2/alexnet'
resnet101 = '../amazon2/resnet101'
resnet34 = '../amazon2/resnet34'

net_list = [
resnet152,
resnet50,
densenet161,
resnet18,
densenet161_new,
densenet201_new,
densenet169_new,
resnet34_new,
resnet50_new,
resnet18_new,
resnet101_new,
vgg11,
alexnet,
resnet101,
resnet34,
]



with open('kdf.pkl', 'rb') as f:
    kfold = pickle.load(f, encoding='latin1')

# labels = []
# for fold in range(5):
#     fold_label_list = []
#     dset_validate = AmazonDateset_validate(kfold[fold][1], IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH, transform_type=0,resize=224)
#     validate_loader = DataLoader(dset_validate, batch_size=128, num_workers=6)
#     for step, (data, target) in enumerate(validate_loader):
#         # data = Variable(data,volatile=True).cuda(validate_gpu)
#         # output = model(data)
#         fold_label_list.append(target)
#     fold_label = torch.cat(fold_label_list)
#     labels.append(fold_label.numpy())
# with open('labels_fold_np','wb') as f:
#     pickle.dump(labels,f)

# print(kfold[0][1])
train_num =0
for i in range(5):
    train_num += len(kfold[i][1])
# sort_labels = np.zeros((train_num,17))
# with open('labels_fold_np','rb') as f:
#     labels_list = pickle.load(f)
# for i in range(5):
#     for id,index in enumerate(kfold[i][1]):
#         # print(type(index))
#         # print(i,id,index)
#         sort_labels[index,:] = labels_list[i][id,:]
#
# for i in range(train_num):
#     if np.sum(sort_labels[i,:]) == 0:
#         print(i)
# with open('sorf_labels.pkl','wb') as f:
#     pickle.dump(sort_labels,f)

# with open('sorf_labels.pkl','rb') as f:
#     sort_labels = pickle.load(f)
l_train = np.load('/home/jianglibin/PythonProject/amazon2/resnet152_1499573211_train.npy')
print(l_train.shape)

def check_validation(model_path):
    v = np.load(os.path.join(model_path,model_path.split('/')[-1]+'_train_lb.npy'))
    # print(v.shape)
    # r = np.mean(v,1)
    # print(r.shape)
    loss = F.binary_cross_entropy(Variable(torch.from_numpy(v)),Variable(torch.from_numpy(l_train)))#in tar
    print(model_path.split('/')[-1]+'_train_lb.npy : ',loss.data[0])

l_test = np.load('/home/jianglibin/PythonProject/amazon2/resnet152_1499573211_test.npy')
# l = np.mean(l,1)
# l = np.mean(l,1)
print(l_test.shape)

def check_predict(model_path):
    p = np.load(os.path.join(model_path,model_path.split('/')[-1]+'_test_lb.npy'))
    # print(os.path.join(model_path,model_path.split('/')[-1]+'_test_lb.npy'))

    # print(p.shape)
    # p = np.mean(p,1)
    # p = np.mean(p,1)
    # print(p.shape)
    loss = F.binary_cross_entropy(Variable(torch.from_numpy(p)),Variable(torch.from_numpy(l_test)))#in tar
    print(model_path.split('/')[-1]+'_test_lb.npy : ',loss.data[0])
    # F.binary_cross_entropy()
    return loss.data[0]


d = {}
for path in net_list:
    model_loss = check_predict(path)
    check_validation(path)
    # d[path.split('/')[-1]] = model_loss

# ll = sorted(d.items(),key=lambda x:x[1])
# print(ll)
# for k,v in ll:
#     print(k,v)
# check_predict(resnet101)

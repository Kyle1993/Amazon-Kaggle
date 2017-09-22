import numpy as np
import pickle
import os
import pandas as pd

# res34_path = '../amazon2/resnet34'
# res101_path = '../amazon2/resnet101'
# alexnet_path = '../amazon2/alexnet'
# squeezenet_path = '../amazon2/squeezenet11'

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
vgg11_path = '../amazon2/vgg11'


test_num = 61191

with open('kdf.pkl', 'rb') as f:
    kfold = pickle.load(f, encoding='latin1')

t = pd.read_csv('sample_submission_v2.csv')
names_sort = list(t['image_name'])
# print(names_sort[:10])


num = 0
index_list = []
for i in range(5):
    num += len(kfold[i][1])
    # print(len(kfold[i][1]))
    index_list.extend(kfold[i][1])
# print(num)res_i = np.zeros((40479, 17))
# print(len(index_list))
# print('---------------')
# print(name_list)





def process_validate_data(path):
    with open(os.path.join(path,'validation_middleoutput_np.pkl'),'rb') as f:
        data = pickle.load(f)

    res = []
    for i in range(8):
        driection = []
        res_i = np.zeros((40479, 17))
        for j in range(5):
            driection.append(data[j][i])
            driection_i = np.vstack(driection)
        # print(driection_i.shape)
        for id in range(40479):
            res_i[index_list[id],:] = driection_i[id,:]
        res.append(res_i)
    res = np.asarray(res,dtype=float)
    res = np.transpose(res,[1,0,2])
    # print(res.shape)
    np.save(os.path.join(path,path.split('/')[-1]+'_train_lb.npy'),res)

def process_test_data(path):
    with open(os.path.join(path,'predict_np.pkl'),'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(path,'name_list.pkl'),'rb') as f:
        name_list = pickle.load(f)

    name_list_d = {}
    for i, v in enumerate(name_list):
        name_list_d[v] = i

    res = []
    for i in range(8):
        d1 = []
        for j in range(5):
            res_j = np.zeros((test_num, 17))
            p = data[j][i]
            for id in range(test_num):
                res_j[id,:] = p[name_list_d[names_sort[id]],:]
            d1.append(res_j)
        res_i = np.asarray(d1)
        res.append(res_i)
    t_res = np.asarray(res)
    t_res = np.transpose(t_res,[2,1,0,3])
    print(t_res.shape)
    np.save(os.path.join(path,path.split('/')[-1]+'_test_lb.npy'),t_res)


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
vgg11_path,
]
for i in range(len(net_list)):
    process_validate_data(net_list[i])
    print('%s validation finish'%net_list[i].split('/')[-1])
    process_test_data(net_list[i])
    print('%s test finish' % net_list[i].split('/')[-1])


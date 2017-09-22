import random
import numpy
import pickle
import torch
from torch.autograd import Variable
import os
from score_thresholds import *


train_num = 40475
test_num = 61191

resnet18_path = '../amazon2/resnet18'
resnet34_path = '../amazon2/resnet34'
alexnet_path = '../amazon2/alexnet'
densenet161_path = '../amazon2/densenet161'
pynet10_path = '../amazon2/pynet10'

paths = [resnet18_path,resnet34_path,alexnet_path,densenet161_path,pynet10_path]
pb_scores = [0.93023,0.93283,0.92744,0.93269,0.91543]
pb_scores = [(s-0.915)*1000 for s in pb_scores]
# pb_scores = [1,1,1]

total_train = numpy.zeros((train_num,17))
total_test = numpy.zeros((test_num,17))

for i in range(len(paths)):
    with open(os.path.join(paths[i],'validation_data_np.pkl'),'rb') as f:
        res = pickle.load(f)
    with open(os.path.join(paths[i],'mean_predict_np.pkl'),'rb') as f:
        test = pickle.load(f)
    total_train += res*pb_scores[i]
    total_test += test*pb_scores[i]

total_train = total_train/sum(pb_scores)
total_test = total_test/sum(pb_scores)

with open(os.path.join(resnet34_path,'validation_labels_np.pkl'),'rb') as f:
    train_labels = pickle.load(f)
with open(os.path.join(resnet34_path,'name_list.pkl'),'rb') as f:
    names = pickle.load(f)

best_th = optimise_f2_thresholds(train_labels,total_train)
score = getScore(train_labels,total_train,best_th)
print(score)


classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
               'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']


print('Writting File...')
with open('result_weight_bag5','w') as f:
    f.write('image_name,tags\n')
    for id in range(total_test.shape[0]):
        s = ''
        for i,v in enumerate(total_test[id]):
            if v>=best_th[i]:
                s += classes[i]+' '
        s = names[id]+','+s+'\n'
        f.write(s)


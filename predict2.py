import pickle
import torch
import os
import numpy as np


model_path = '../amazon2/resnet34'
test_num = 61191

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
           'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
           'primary', 'road', 'selective_logging', 'slash_burn', 'water']

with open(os.path.join(model_path,'best_th.pkl'),'rb') as f:
    threshold = pickle.load(f)
with open(os.path.join(model_path,'name_list.pkl'),'rb') as f:
    names = pickle.load(f)
with open(os.path.join(model_path,'loss_info.pkl'),'rb') as f:
    loss_info = pickle.load(f)


total_res  = np.zeros((test_num,17))
k=5
total_loss = 0
for i in range(k):
    with open(os.path.join(model_path,'fold%d_predict_np.pkl'%i),'rb') as f:
        total_res += pickle.load(f)*(1-loss_info[i])
    total_loss += (1-loss_info[i])

total_res = total_res/total_loss


print('Writting File...')
with open(os.path.join(model_path,'result_weight'),'w') as f:
    f.write('image_name,tags\n')
    print(total_res.shape[0])
    for id in range(total_res.shape[0]):
        s = ''
        for i,v in enumerate(total_res[id]):
            if v>=threshold[i]:
                s += classes[i]+' '
        s = names[id]+','+s+'\n'
        f.write(s)
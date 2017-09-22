import pickle
import numpy as np
import os

# model_path = '../amazon2/_alexnet'
#
# with open(os.path.join(model_path, 'validation_globals_output_np.pkl'), 'rb') as f:
#     globals_output = pickle.load(f)
# with open(os.path.join(model_path, 'validation_globals_labels_np.pkl'), 'rb') as f:
#     globals_labels = pickle.load(f)
# with open(os.path.join(model_path, 'validation_labels_np,pkl'), 'rb') as f:
#     labels_np = pickle.load(f)
# with open(os.path.join(model_path, 'validation_middleoutput_np.pkl'), 'rb') as f:
#     eval_np = pickle.load(f)
#
# with open(os.path.join(model_path, 'predict_np.pkl'), 'rb') as f:
#     predict_np = pickle.load(f)
#
#
# print(32384+8095,(32384+8095)*8,8095*5*8)
# print('-------------------')
# print(globals_output.shape)
# print(globals_labels.shape)
# print(labels_np.shape)
# print(len(eval_np),len(eval_np[0]))
# print(eval_np[-1])
# print(len(predict_np),len(predict_np[0]))
# print(predict_np[0][0].shape)
#
# print('-----------')
# for i in range(5):
#     for t  in range(8):
#         print(eval_np[i][t].shape)

model_path = '../amazon2/vgg11'
fold = 3

with open(os.path.join(model_path,'eval_fold%d_np.pkl'%fold),'rb') as f:
    eval_np = pickle.load(f)
with open(os.path.join(model_path,'predict_fold%d_np.pkl'%fold),'rb') as f:
    predict_np = pickle.load(f)
with open(os.path.join(model_path,'name_list.pkl'),'rb') as f:
    names = pickle.load(f)

print(len(eval_np))
for i in range(8):
    print(eval_np[i].shape)
    print(type(eval_np[i]))
print(len(predict_np))
for i in range(8):
    print(predict_np[i].shape)
    print(type(predict_np[i]))
# print(names)
# print(type(names))
print(len(names))

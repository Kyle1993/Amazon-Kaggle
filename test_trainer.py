import torch
from torch.autograd import Variable
import argparse
import copy
import pickle
from Datahelper2 import *
from Model import *
from torch.utils.trainer import trainer

train_num = 32384
validate_num = 8095
DS = '/home/jianglibin/PythonProject/Amazon/train_validate_dataset.h5'
IMG_TRAIN_PATH = '/home/jianglibin/PythonProject/Amazon/data/train-jpg/'
LABEL_PATH = '/home/jianglibin/PythonProject/amazon2/labels.pkl'
IMG_EXT = '.jpg'

with open('kdf.pkl', 'rb') as f:
    kfold = pickle.load(f, encoding='latin1')
train_index = kfold[0][0]

k = 5
epochs = 2
# lr = 1e-4
weight_decay = 0
momentum = 0
dset_train = AmazonDateset_train(train_index, IMG_TRAIN_PATH, IMG_EXT, LABEL_PATH, resize=256)
train_loader = DataLoader(dset_train, batch_size=2, shuffle=True, num_workers=4)
model = AM_resnet18()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mytrainer = trainer.Trainer(model,criterion,optimizer,train_loader,1)
mytrainer.run(2)

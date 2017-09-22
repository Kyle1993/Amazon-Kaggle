import pandas as pd
from torch import np # Torch wrapper for Numpy

import os
from PIL import Image
import h5py
import random
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle

class AmazonDateset_train(Dataset):
    def __init__(self, train_index, img_path, img_ext,label_path,resize=None):
        super(AmazonDateset_train, self).__init__()
        self.img_path = img_path
        self.img_ext = img_ext
        if resize != 256:
            self.transform = transforms.Compose([transforms.Scale(resize),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


        self.img_index = train_index
        self.label = pickle.load(open(label_path,'rb'))


    def __getitem__(self, index):
        img_index = index//8
        tft = index%8
        img = Image.open(self.img_path + 'train_'+str(self.img_index[img_index]) + self.img_ext)
        if tft >= 4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        r = tft % 4
        R = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][r]
        if R != None:
            img = img.transpose(R)

        img = img.convert('RGB')
        img = self.transform(img)
        label = torch.from_numpy(self.label['train_'+str(self.img_index[img_index])]).float()
        return img, label

    def __len__(self):
        return len(self.img_index)*8


class AmazonDateset_validate(Dataset):
    def __init__(self, validate_index, img_path, img_ext,label_path,transform_type=0,random_transform=False,resize=None):
        super(AmazonDateset_validate, self).__init__()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform_type = transform_type
        self.random_transform = random_transform
        if resize != 256:
            self.transform = transforms.Compose([transforms.Scale(resize),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        self.img_index = validate_index
        self.label = pickle.load(open(label_path,'rb'))

    def __getitem__(self, index):
        img = Image.open(self.img_path + 'train_'+str(self.img_index[index]) + self.img_ext)
        if self.random_transform:
            tft = random.randint(0, 7)
        else:
            tft = self.transform_type
        if tft >= 4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        r = tft % 4
        R = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][r]
        if R != None:
            img = img.transpose(R)

        img = img.convert('RGB')
        img = self.transform(img)
        label = torch.from_numpy(self.label['train_'+str(self.img_index[index])]).float()
        return img, label

    def __len__(self):
        return len(self.img_index)

class KaggleAmazonDataset_test(Dataset):

    def __init__(self, img_path,transform_type=0,resize=None):

        self.img_dir = img_path
        self.img_list = os.listdir(img_path)
        self.transform_type = transform_type
        if resize != 256:
            self.transform = transforms.Compose([transforms.Scale(resize),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img = Image.open(self.img_dir + self.img_list[index])

        tft = self.transform_type  # transform_type
        if tft >= 4:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        r = tft % 4
        R = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][r]
        if R != None:
            img = img.transpose(R)

        img = img.convert('RGB')
        img = self.transform(img)

        return img,self.img_list[index].split('.')[0]

    def __len__(self):
        return len(self.img_list)




if __name__=='__main__':
    DS = '/home/kyle/PythonProject/Amazon/train_validate_dataset.h5'
    # IMG_TRAIN_PATH = '/home/jianglibin/PythonProject/Amazon/data/train-jpg/'
    CSV_PATH = '/home/kyle/PythonProject/AmazonData/train_v2.csv'
    IMG_PATH = '/home/kyle/PythonProject/AmazonData/train-jpg/'
    IMG_EXT = '.jpg'
    LABEL_PATH = '/home/kyle/PythonProject/Amazon/labels.h5'

    IMG_TEST_PATH = '/home/kyle/PythonProject/AmazonData/test-jpg/'



import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os

class CDataset(Dataset):
    def __init__(self, train=True):    
        self.df = pd.read_csv('/opt/ml/input/data/train/label.csv')
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        X = Image.open('/opt/ml/input/data/train/images/' + self.df.loc[idx]['path'] + '/' + self.df.loc[idx]['fname'])
        #X = np.array(X)
        #X = cv2.cvtColor(X,cv2.COLOR_RGBA2RGB)
        #xKer = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        #yker = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        #xedge = cv2.filter2D(X,-1,xKer)
        #yedge = cv2.filter2D(X,-1,yker)
        #X = xedge + yedge
        #print(f'X Shape1: {X.shape}')
        X = transforms.ToTensor()(X)
        #print(f'X Shape2: {X.shape}')
        #X = X.permute(1,2,0)
        #print(f'X Shape3: {X.shape}')
        y = self.df.loc[idx]['label']

        #X = transforms.ToTensor()(X)
        y = torch.tensor(y)

        return X,y


class Submission_Dataset(Dataset):
    def __init__(self,subm_path):
        self.subm_path = subm_path
        self.df = pd.read_csv(os.path.join(self.subm_path,'info.csv'))
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        X = Image.open(self.subm_path + '/images/' + self.df.loc[idx]['ImageID'])
        #X = np.array(X)
        #X = cv2.cvtColor(X,cv2.COLOR_RGBA2RGB)
        #xKer = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        #yker = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        #xedge = cv2.filter2D(X,-1,xKer)
        #yedge = cv2.filter2D(X,-1,yker)
        #X = xedge + yedge
        #print(f'X Shape1: {X.shape}')
        X = transforms.ToTensor()(X)
        return X

class pyTest():
    def __init__(self):
        self.t = 50
    def print_test(self,x):
        #print(f'it is ??')
        return x

# if __name__ == '__main__':
#     #temp = CDataset(train=True)
#     #d = pd.read_csv('/opt/ml/input/data/train/label.csv')
#     #img = Image.open('/opt/ml/input/data/train/images/' + d.loc[0]['path'] + '/' + d.loc[0]['fname'])
#     #print(img.shape)
#     d_set = CDataset(train=True)
#     dd, dl = next(iter(d_set))
#     print(dd.shape)    
    


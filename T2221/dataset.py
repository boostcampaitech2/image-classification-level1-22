from pandas.io.pytables import IncompatibilityWarning
import numpy as np
import os
import random

import torch
from torchvision.transforms import transforms
from PIL import Image
import warnings
import pandas as pd
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler


warnings.filterwarnings(action='ignore')


class Image_Dataset(Dataset):
    def __init__(self, data, transform= None,train= True):
        self.data= data
        self.train= train
        self.transform= transform

        if self.train:
            self.get_labels= self.data.loc[:, 'label'].tolist

        # if self.train:
        #     self.get_labels= self.data.loc[:, 'label'].tolist()
        #     print(self.get_labels)

    def __getitem__(self, idx):
        if self.train:
            image= Image.open(self.data.iloc[idx]['path'])
            label= self.data.iloc[idx]['label']

            if self.transform:
                image= self.transform(image)
    
            return image, label

        else:
            image = Image.open(self.data[idx])

            if self.transform:
                image = self.transform(image)

            return image

    def __len__(self):
        if self.train:
            return len(self.data['label'])

        else: return len(self.data)


if __name__ == '__main__':
    CSV_PATH= '/opt/ml/code/train_with_label2.csv'
    data= pd.read_csv(CSV_PATH)

    image_data= Image_Dataset(data,
                              transform= transforms.Compose([
                                  transforms.ToTensor()
                              ]),
                              train= True)
    print(image_data.get_labels)
    # img_loader= torch.utils.data.DataLoader(image_data, batch_size= 64, shuffle= True)
    # train_loader= DataLoader(image_data, batch_size= 64, sampler= ImbalancedDatasetSampler(image_data))
    # print(next(iter(image_data[0])).shape)

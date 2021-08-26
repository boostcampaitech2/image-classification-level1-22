import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from matplotlib import pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, csv_file, drop_features=[], transform=None, train=True):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(columns=drop_features)
        self.X = self._read_images(self.data['path']) # image
        self.y = self._read_labels(self.data['label']) # label
        self.features = list(self.data.drop(columns=['path', 'label']))
        self.classes = [i for i in range(18)]
        self.train = train
        self.transform = transform
        print("Dataset has been loaded")
    
    def _read_images(self, paths):
        images = list()
        for file in tqdm(paths):
            img = Image.open(file)
            images.append(img)
        return images
    
    def _read_labels(self, labels):
        return labels.tolist()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X, y = None, None
        
        X = self.X[idx]
        if self.transform:
            X = self.transform(X)
            
        if self.train:
            y = np.int64(self.y[idx])
            # y = self.y[idx]
        
        return X, y
    
    def get_image(self, idx):
        X = self.X[idx]
        return X
    

if __name__ == '__main__':
    train_with_label = '/opt/ml/input/data/train/train_with_label.csv'
    custom_dataset_train = CustomDataset(train_with_label,
                                         drop_features=['id', 'race'],
                                        transform=transforms.Compose([
                                            transforms.ToTensor()
                                        ]),
                                        train=True)
    image, label = next(iter(custom_dataset_train))
    print(custom_dataset_train[0])
    # print(type(image), image.shape)
    # print(type(label), label)
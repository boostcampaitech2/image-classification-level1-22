import pandas as pd
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *   
import albumentations as A     
    

class CustomDataset(Dataset):
    def __init__(self, csv_file, val_ratio=0.2, transform=None, train=True, test=True):
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['path']
        self.class_label = self.data['label']

        self.val_ratio = val_ratio
        self.train = train
        self.transform = transform
        self.test = test
        if not self.train:
            self.set_random_seed(999)

    def set_random_seed(self, SEED=999):
        np.random.seed(SEED)
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    def _read_images(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        X = self._read_images(index)
            
        if self.transform:
            X = self.transform(X)
            
        if self.train:
            y = self.class_label[index]
        
        return X, y

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def _get_min_class_num(self):
        min_class_num = min(self.class_label.value_counts())
        return min_class_num

    def split_balanced_dataset(self):
        min_class_num = self._get_min_class_num()
        labels = set(self.class_label)
        total_indices = []

        for label in labels:
            indices = self.class_label == label
            indices = indices.index[indices]
            total_indices += random.choices(indices, k=min_class_num)

        _train_set, val_set = self.split_dataset()

        return [Subset(self, total_indices), val_set]


class AlbumentationDataset(CustomDataset):
    def _read_images(self, index):
        return super()._read_images(index)

    def __getitem__(self, index):
        X = self._read_images(index)
        X = np.array(X)

        # if self.transform:
        transform = A.Compose([self.transform, A.pytorch.ToTensorV2()])
        X = transform(image=X)['image']
            
        if self.train:
            y = self.class_label[index]
        
        X = X / 255.0
        return X, y

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

    

if __name__ == '__main__':
    
    train_with_label = '/opt/ml/input/data/train/train_with_label.csv'
    transform_list = transforms.Compose([transforms.ToTensor()])
    custom_dataset_train = CustomDataset(train_with_label,
                                        transform=transform_list,
                                        train=True)

    image, label = next(iter(custom_dataset_train))
    print(type(image.to('cuda')), type(label))


    transform_list = A.Compose([A.HorizontalFlip(p=1)])

    custom_dataset_train = AlbumentationDataset(train_with_label,
                                        transform=transform_list,
                                        train=True)

    image, label = next(iter(custom_dataset_train))
    print(type(image), type(label))

    balanced_train_set, valid_set = custom_dataset_train.split_balanced_dataset()
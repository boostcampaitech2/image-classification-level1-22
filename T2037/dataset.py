import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms


class Mask_Dataset(Dataset):
    """Some Information about MyDataset"""

    def __init__(self, path, transform=None, train=True):
        super(Mask_Dataset, self).__init__()
        self.data = pd.read_csv(path)
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        X = Image.open(self.data.iloc[idx]["path"])

        if self.transform:
            X = self.transform(X)

        if self.train:
            label = self.data.iloc[idx]["label"]
            return X, label

        return X

    def __len__(self):
        return len(self.data["label"])

class Mask_Dataset_Age(Dataset):
    """오로지 나이만을 판별하는 모델 전용 데이터셋"""
    def __init__(self, path, transform=None, train=True):
        super(Mask_Dataset_Age, self).__init__()
        self.data = pd.read_csv(path)
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        X = Image.open(self.data.iloc[idx]["path"])

        if self.transform:
            X = self.transform(X)

        if self.train:
            # 30세 이하 0, 중간 1, 60세 이상 2 
            # 0,3,6... = 30세 이하, 1,4,7...
            label = self.data.iloc[idx]["label"] % 3
            return X, label

        return X

    def __len__(self):
        return len(self.data["label"])

class Mask_Dataset_Jender(Dataset):
    """오로지 성별만을 판별하는 모델 전용 데이터셋"""
    def __init__(self, path, transform=None, train=True):
        super(Mask_Dataset_Jender, self).__init__()
        self.data = pd.read_csv(path)
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        X = Image.open(self.data.iloc[idx]["path"])

        if self.transform:
            X = self.transform(X)

        if self.train:
            # 남성 0, 여성 1
            # 012 / 345 / 678 / .....
            label = (self.data.iloc[idx]["label"] // 3) % 2
            return X, label

        return X

    def __len__(self):
        return len(self.data["label"])

class Mask_Dataset_Mask(Dataset):
    """오로지 마스크 찾용 여부만을 판별하는 모델 전용 데이터셋"""
    def __init__(self, path, transform=None, train=True):
        super(Mask_Dataset_Mask, self).__init__()
        self.data = pd.read_csv(path)
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        X = Image.open(self.data.iloc[idx]["path"])

        if self.transform:
            X = self.transform(X)

        if self.train:
            # 썼으면 0, 잘못쓰면 1, 안쓰면 2
            label = (self.data.iloc[idx]["label"] // 6)
            return X, label

        return X

    def __len__(self):
        return len(self.data["label"])


if __name__ == "__main__":
    CSV_PATH = "/opt/ml/code/splitted_train.csv"
    image_data = Mask_Dataset(
        CSV_PATH,
        transform=transforms.Compose([transforms.ToTensor()]),
        train=True,
    )
    print(len(image_data))
    print(next(iter(image_data)))

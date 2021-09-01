import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

class SimpleAugmentation:
    def __init__(self, **args):
        self.transform = transforms.Compose([
            ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image)

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

####################################

def get_mask(mask):
    if 'incorrect' in mask:
        return 'incorrect'
    elif 'normal' in mask:
        return 'normal'
    else:
        return 'mask'

def get_age(age):
    if age < 30:
        return 'young'
    elif age < 58: # 58, 59 세도 60세 이상으로 분류
        return 'middle'
    else:
        return 'old'

def get_label(mask, gender, age):
    weights = {
        'mask': {'mask': 0, 'incorrect': 6, 'normal': 12},
        'gender': {'male': 0, 'female': 3},
        'age': {'young': 0, 'middle': 1, 'old': 2}
    }
    return weights['mask'][mask] + weights['gender'][gender] + weights['age'][age]

class CustomDataset(Dataset):
    num_classes = 3 * 2 * 3

    image_paths = []
    labels = []
    indices = dict(train=[], val=[])

    def __init__(self, data_dir, val_ratio, seed, mean=None, std=None):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.seed = seed
        self.mean = mean
        self.std = std

        self.mode = 'train' if 'train' in self.data_dir else 'eval'
        self.transform = None

        if self.mode == 'train':
            self.train_setup()
        elif self.mode == 'eval':
            self.eval_setup()

        self.calc_statistics()

    def train_setup(self):
        self.df_train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))

        self.folders = os.listdir(os.path.join(self.data_dir, 'images'))
        self.folders = [folder for folder in self.folders if not folder.startswith('._')]

        split_folders = dict(train=[], val=[])
        if self.val_ratio == 0:
            split_folders['train'] = self.folders
            split_folders['val'] = list()
        else:
            train_folders, val_folders = train_test_split(self.folders, test_size=self.val_ratio, random_state=self.seed)
            split_folders['train'] = train_folders
            split_folders['val'] = val_folders         

        idx = 0
        for _, row in self.df_train.iterrows():
            images = os.listdir(os.path.join(self.data_dir, 'images', row['path']))
            images = sorted([image for image in images if not image.startswith('._') and image != '.ipynb_checkpoints'])

            for image in images:
                self.image_paths.append(os.path.join(self.data_dir, 'images', row['path'], image))
                self.labels.append(get_label(get_mask(image), row['gender'], get_age(row['age'])))

                for phase in split_folders.keys():
                    if row['path'] in split_folders[phase]:
                        self.indices[phase].append(idx)
                idx += 1

    def eval_setup(self):
        self.df_eval = pd.read_csv(os.path.join(self.data_dir, 'info.csv'))

        for _, row in self.df_eval.iterrows():
            self.image_paths.append(os.path.join(self.data_dir, 'images', row['ImageID']))

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in tqdm(self.image_paths):
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            return image, self.labels[idx]
        elif self.mode == 'eval':
            return image

    def split_dataset(self):
        if self.mode == 'train':
            return [Subset(self, indices) for phase, indices in self.indices.items()]
        elif self.mode == 'eval':
            raise ValueError(f"train 시에만 split 이 가능합니다, {self.mode}")
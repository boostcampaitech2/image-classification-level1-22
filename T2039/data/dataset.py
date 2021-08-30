import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from PIL import Image

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
    elif age < 60:
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

    def __init__(self, data_dir, transform, val_ratio, seed):
        self.data_dir = data_dir
        self.mode = 'train' if 'train' in self.data_dir else 'eval'
        self.transform = transform
        self.val_ratio = val_ratio
        self.seed = seed

        if self.mode == 'train':
            self.train_setup()
        elif self.mode == 'eval':
            self.eval_setup()

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
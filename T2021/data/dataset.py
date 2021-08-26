import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class MaskLabels:
  mask = 0
  incorrect = 1
  normal = 2

class GenderLabels:
  male = 0
  female = 1

class AgeGroup:
  map_label = lambda x:0 if int(x)<30  else 1 if int(x) <60 else 2

class MaskBaseDataset(Dataset):
  num_classes = 18

  _file_names = {
    "mask1.jpg": MaskLabels.mask,
    "mask2.jpg": MaskLabels.mask,
    "mask3.jpg": MaskLabels.mask,
    "mask4.jpg": MaskLabels.mask,
    "mask5.jpg": MaskLabels.mask,
    "incorrect_mask.jpg": MaskLabels.incorrect,
    "normal.jpg": MaskLabels.normal
  }

  image_paths = []
  mask_labels = []
  gender_labels = []
  age_labels = []
  multi_labels = []

  def __init__(self, img_dir, transform = None):
    self.img_dir = img_dir
    self.transform = transform

    self.setup()

  def self_transform(self, transform):
    self.transform = transform

  def setup(self):
    profiles = os.listdir(self.img_dir)
    for profile in profiles:
      for file_name, label in self._file_names.items():
        img_path = os.path.join(self.img_dir, profile, file_name)
        if os.path.exists(img_path):
          self.image_paths.append(img_path)
          self.mask_labels.append(label)
          
          id, gender, race, age = profile.split("_")
          gender_label = getattr(GenderLabels, gender)
          age_label = AgeGroup.map_label(age)

          self.gender_labels.append(gender_label)
          self.age_labels.append(age_label)

  def __getitem__(self, index):
      image_path = self.image_paths[index]
      image = Image.open(image_path)

      mask_label = self.mask_labels[index]
      gender_label = self.gender_labels[index]
      age_label = self.age_labels[index]
      multi_class_label = mask_label * 6 + gender_label * 3 + age_label
      self.multi_labels.append(multi_class_label)

      image_transform = self.transform(image)
      
      return image_transform, multi_class_label

  def __len__(self):
    return len(self.image_paths)

class MaskDataset(Dataset):
  def __init__(self, csv_file, img_path, transform, multi_label = False):
    self.csv_file = csv_file
    self.img_path = img_path
    self.transform = transform
    self.multi_label = multi_label

  def __len__(self):
    return len(self.csv_file)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_name = os.path.join(self.img_path, self.csv_file["img_path"][idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    if self.multi_label:
      label = self.csv_file[["mask", "gender", "age"]].iloc[idx].values
    else:
      label = self.csv_file["label"].iloc[idx]

    return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

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

class get_pretext_dataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.csv_file)
  
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.dataset[idx]
        label = label % 3

        return img, label

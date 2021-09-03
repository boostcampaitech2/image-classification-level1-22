import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class MakeLabel:
    def __init__(self, BASE_PATH, val_ratio):
        self.base_path= BASE_PATH
        self.csv_path= '/opt/ml/code2/valset.csv'
        self.img_path= os.path.join(self.base_path, 'images')

        self.val_ratio= val_ratio
        self.cols= ['mask', 'gender', 'age', 'mask_label', 'gender_label', 'age_label', 'path', 'label']
        
        self.train_df= pd.DataFrame(columns= self.cols)
        self.val_df= pd.DataFrame(columns= self.cols)


    def split_profile(self, profile_list, val_ratio):
        train_profiles, val_profiles= train_test_split(profile_list, test_size= self.val_ratio, stratify= profile_list['age'],random_state= 42)

        return train_profiles, val_profiles

    def get_gender_label(self, df, idx):
        if df.loc[idx, 'gender']=='male': return 0
        else: return 1

    def get_age_label(self, df, idx):
        if df.loc[idx, 'age']< 30: return 0
        elif df.loc[idx, 'age']< 58: return 1
        else: return 2

    def get_mask_label(self, df, idx):
        mask= df.loc[idx, 'mask'].lower()
        if mask== 'normal': return 2
        elif mask== 'incorrect_mask': return 1
        else: return 0

    def labeling(self, data, df):
        # print(data)
        
        idx_df= 0
        for idx in tqdm(range(data.shape[0])):
            img_dir= os.path.join(self.img_path, data.iloc[idx]['path'])
            for img in os.listdir(img_dir):
                if img.startswith('._'): continue

                img_name, ext= os.path.splitext(img)
                df.loc[idx_df, ['gender', 'age']]= data.iloc[idx][['gender', 'age']]
                df.loc[idx_df, ['path']]= os.path.join(img_dir, img)
                df.loc[idx_df, ['gender_label']]= self.get_gender_label(df, idx_df)
                df.loc[idx_df, ['age_label']]= self.get_age_label(df, idx_df)
                df.loc[idx_df, ['mask']]= img_name
                df.loc[idx_df, ['mask_label']]= self.get_mask_label(df, idx_df)

                df.loc[idx_df, ['label']]= df.loc[idx_df, ['mask_label']][0] *6 + df.loc[idx_df, ['gender_label']][0] *3 \
                    + df.loc[idx_df, ['age_label']][0]
                idx_df+=1
        
        return df
    
    def run(self):
        data= pd.read_csv(self.csv_path)
        
        # trainset, valset= self.split_profile(data, self.val_ratio)
        # print(trainset.shape, valset.shape)

        trainset= data
        self.train_df= self.labeling(trainset, self.train_df)
        # self.val_df= self.labeling(valset, self.val_dfx)
        # print(self.train_df.shape)

        self.train_df.to_csv('/opt/ml/code2/trainset_LABEL.csv')
        # self.val_df.to_csv('/opt/ml/code2/val_label_58_st.csv')
        
class TrainDataset(Dataset):
    def __init__(self, csv_path, transforms= None, train= True):
        self.data= pd.read_csv(csv_path)
        self.transforms= transforms
        self.get_labels= self.data.loc[:, 'label'].tolist
    
    def __getitem__(self, idx):
        image= Image.open(self.data.iloc[idx]['path']) # csv 안의 path에 있는 경로로 이미지 Load
        label= self.data.iloc[idx]['label']

        if self.transforms:
            image= self.transforms(image)
        
        return image, label

    def __len__(self):
        return (len(self.data['label']))

class TestDataset(Dataset):
    def __init__(self, csv_path, base_path, transforms= None):
        self.data= pd.read_csv(csv_path)
        self.img_path= os.path.join(base_path, 'images')
        self.transforms= transforms
    
    def __getitem__(self, idx):
        image= Image.open(os.path.join(self.img_path, self.data.iloc[idx]['ImageID']))

        if self.transforms:
            image= self.transforms(image)

        return image
    
    def __len__(self):
        return len(self.data['ans'])

if __name__ == '__main__':
    BASE_PATH= '/opt/ml/input/data/train'
    val_ratio= 0.2
    
    makelabel= MakeLabel(BASE_PATH, val_ratio)
    makelabel.run()
    
    # TEST_CSV_PATH= '/opt/ml/input/data/eval/info.csv'
    # TEST_BASE_PATH= '/opt/ml/input/data/eval/'
    
    # test= TestDataset(TEST_CSV_PATH, TEST_BASE_PATH)
    # print(next(iter(test)))

    # data= pd.read_csv('/opt/ml/code2/csv/all_.csv')
    # older= TrainDataset('/opt/ml/code2/csv/train_old_58.csv',
    # transforms= transforms.Compose([
    #     transforms.ToTensor()
    # ]))
    # # print(older[1])
    # print(len(older))
    # # print(older[0])

    # np.random.seed(42)
    # # random.seed(42)
    # # lam = np.random.beta(1, 1)
    # # print(lam)
    # idx= random.randrange(0, len(older))
    # # img, lab= older[idx]
    # print(torch.tensor(older[idx][0]))
    



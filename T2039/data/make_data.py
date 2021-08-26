import numpy as np
import pandas as pd
import os
from tqdm import tqdm

INPUT_DATA_ROOT_PATH = '/opt/ml/input/data'

class TrainData:
    def __init__(self, age_cat_type=1):
        """
        type
            1: 원래 조건 대로 데이터 생성
            2: 58,59세를 60세 카테고리에 포함시켜 데이터 생성
        """
        self.age_cat_type = age_cat_type
        self.TRAIN_PATH = os.path.join(INPUT_DATA_ROOT_PATH, 'train')
        self.TRAIN_IMAGE_PATH = os.path.join(self.TRAIN_PATH, 'images')
        self.df_train = pd.read_csv(os.path.join(self.TRAIN_PATH, 'train.csv'))

        self.make_data()

    def get_age_cat(self, x):
        return '~30' if x < 30 else ('60~' if x >= 60 else '30~60')

    def get_age_cat2(self, x):
        return '~30' if x < 30 else ('60~' if x >= 58 else '30~60')

    def get_mask(self, x):
        return 'incorrect' if 'incorrect_mask' in x else ('not wear' if 'normal' in x else 'wear')

    def get_class(self, gender, age_cat, mask):
        w_age_cat = {'~30': 0, '30~60': 1, '60~': 2}
        w_gender = {'male': 0, 'female': 3}
        w_mask = {'wear': 0, 'incorrect': 6, 'not wear': 12}
        return w_age_cat[age_cat] + w_gender[gender] + w_mask[mask]

    def make_data(self):
        new_data = {'id': [], 'gender': [], 'age': [], 'age_cat': [], 'mask': [], 'path': [], 'file': [], 'class': []}
        org_cols = ['id', 'gender', 'age', 'path']

        for i, row in self.df_train.iterrows():
            file_list = os.listdir(os.path.join(self.TRAIN_IMAGE_PATH, row.path))
            file_list = [file for file in file_list if not file.startswith('._') and file != '.ipynb_checkpoints']
            file_list.sort()
            
            for f_name in file_list:
                for org_col in org_cols:
                    new_data[org_col].append(row[org_col])
                
                if self.age_cat_type == 1:
                    age_cat = self.get_age_cat(row.age)
                elif self.age_cat_type == 2:
                    age_cat = self.get_age_cat2(row.age)
                mask = self.get_mask(f_name)
                new_data['age_cat'].append(age_cat)
                new_data['file'].append(f_name)
                new_data['mask'].append(mask)
                new_data['class'].append(self.get_class(row.gender, age_cat, mask))
            
        self.df_train_new = pd.DataFrame(new_data)

    def get_data(self):
        return self.df_train_new

    def save_to_csv(self):
        if self.age_cat_type == 1:
            self.df_train_new.to_csv(os.path.join(self.TRAIN_PATH, 'train_new.csv'), index=False)
        elif self.age_cat_type == 2:
            self.df_train_new.to_csv(os.path.join(self.TRAIN_PATH, 'train_new2.csv'), index=False)

###################################################################################################

if __name__ == '__main__':
    TrainData(2).save_to_csv()
    

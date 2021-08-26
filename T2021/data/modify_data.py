import os
import pandas as pd

train_path = '../input/data/train/'
test_path = '../input/data/eval/'

csv_file = pd.read_csv(os.path.join(train_path, 'train.csv'))

#male: 0, female: 1
#<30: 0, >=30 and <60:1, >=60:2

df_csv = csv_file.copy()

df_csv = df_csv.drop(['id', 'race'], axis = 1)

df_csv.loc[df_csv['gender'] =='male', 'gender'] = 0
df_csv.loc[df_csv['gender'] =='female', 'gender'] = 1

df_csv.loc[df_csv['age'] < 30, 'age'] = 0
df_csv.loc[(df_csv['age'] >=30) & (df_csv['age'] <60), 'age'] = 1
df_csv.loc[df_csv['age'] >= 60, 'age'] = 2

df_list = []

for foldername in os.listdir(os.path.join(train_path, 'images/')):
  if foldername[0]=='.':
    continue

  for file in os.listdir(os.path.join(train_path, 'images/'+foldername)):
    if file[0]=='.':
      continue

    img_path = os.path.join(foldername+'/'+file)
    fsplit = file.split('.')
    
    mask_conditon = 0
    if fsplit[0] == "incorrect_mask": 
      mask_condition = 1
    elif fsplit[0] == "normal":
      mask_condition = 2
    else:
      mask_condition = 0

    df_list.append([foldername, img_path, mask_condition])

df_mask = pd.DataFrame(df_list)
df_mask.columns=['path', 'img_path', 'mask']
df_mask.set_index('path')

train_df = df_csv.join(df_mask.set_index('path'), on='path')

label_list = []

for index, row in train_df.iterrows():
  if row['mask'] == 0:
    if row['gender'] == 0:
      if row['age'] == 0:
        label_list.append(0)
      elif row['age'] == 1:
        label_list.append(1)
      else:
        label_list.append(2)
    if row['gender'] == 1:
      if row['age'] == 0:
        label_list.append(3)
      elif row['age'] == 1:
        label_list.append(4)
      else:
        label_list.append(5)
  if row['mask'] == 1:
    if row['gender'] == 0:
      if row['age'] == 0:
        label_list.append(6)
      elif row['age'] == 1:
        label_list.append(7)
      else:
        label_list.append(8)
    if row['gender'] == 1:
      if row['age'] == 0:
        label_list.append(9)
      elif row['age'] == 1:
        label_list.append(10)
      else:
        label_list.append(11)
  if row['mask'] == 2:
    if row['gender'] == 0:
      if row['age'] == 0:
        label_list.append(12)
      elif row['age'] == 1:
        label_list.append(13)
      else:
        label_list.append(14)
    if row['gender'] == 1:
      if row['age'] == 0:
        label_list.append(15)
      elif row['age'] == 1:
        label_list.append(16)
      else:
        label_list.append(17)


train_df['label'] = label_list
train_df.to_csv(os.path.join(train_path, 'train_modified.csv'), index=False)
print('Created train_modified.csv')
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def group_age_stratify(data):
    print(data.groupby('age').count())
    print(data.isnull().sum())

    trainset, valset= train_test_split(data, test_size= 0.2, stratify= data['age'], random_state= 42)
    
    return trainset, valset

def group_age_balance(data):
    # 나이별로 균등하게 validation set을 만드는 코드 (50명씩 추출)
    ages= [18, 20, 23, 30, 43, 57, 58, 59, 60]
    for age in data.groupby('age').count().index:
#        print(age)
#        print(len(data.loc[data['age']== age]) )
        train, val= train_test_split(data.loc[data['age']== age], test_size= 50/len(data.loc[data['age']== age]), random_state= 42)
        trainset= pd.concat([trainset, train])
        valset= pd.concat([valset, val])
        
    print(trainset.shape, valset.shape)
    
    return trainset, valset

def old_data_from_train(trainset):
    old_data= trainset[trainset['age']>=57]
    
    return old_data


data= pd.read_csv('/opt/ml/input/data/train/train.csv')

data.loc[(18<=data['age']) & (data['age']< 20), 'age']= 18
data.loc[(20<=data['age']) & (data['age']< 23), 'age']= 20
data.loc[(23<=data['age']) & (data['age']< 28), 'age']= 23

data.loc[(28<=data['age']) & (data['age']< 43), 'age']= 30
data.loc[(43<=data['age']) & (data['age']< 53), 'age']= 43
data.loc[(53<=data['age']) & (data['age']< 58), 'age']= 57

data.loc[data['age']== 58, 'age']= 58
data.loc[data['age']== 59, 'age']= 59
data.loc[data['age']== 60, 'age']= 60

trainset, valset= group_age_stratify(data)

print(trainset.groupby('age').count())
print(valset.groupby('age').count())

trainset.to_csv('/opt/ml/code2/trainset.csv')
valset.to_csv('/opt/ml/code2/valset.csv')

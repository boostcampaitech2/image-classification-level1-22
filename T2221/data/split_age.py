import os
import pandas as pd
from sklearn.model_selection import train_test_split

data= pd.read_csv('/opt/ml/input/data/train/train.csv')
trainset= data
# trainset.to_csv('/opt/ml/code2/trainset.csv')

# trainset= pd.DataFrame(columns= ['id', 'gender', 'race', 'path'])
# valset=pd.DataFrame(columns=['id', 'gender', 'race', 'path'])

# ages= [18, 20, 23, 30, 43, 57, 58, 59, 60]
# # 여기서 나눠주기만 하면.. 내가 만든 파일로 train, val을 만들 수 있다.

# print(data.groupby('age').count())
# data.loc[data['age']==19, 'age']= data.loc[data['age']==19, 'age'].sample(n= 307, random_state= 42)
# print(data.groupby('age').count())

data.loc[(18<=data['age']) & (data['age']< 20), 'age']= 18
data.loc[(20<=data['age']) & (data['age']< 23), 'age']= 20
data.loc[(23<=data['age']) & (data['age']< 28), 'age']= 23

data.loc[(28<=data['age']) & (data['age']< 43), 'age']= 30
data.loc[(43<=data['age']) & (data['age']< 53), 'age']= 43
data.loc[(53<=data['age']) & (data['age']< 58), 'age']= 57

data.loc[data['age']== 58, 'age']= 58
data.loc[data['age']== 59, 'age']= 59
data.loc[data['age']== 60, 'age']= 60

# data.loc[data['age']== 19, 'age']
# print(data.groupby('age').count())
# trainset= data

print(data.groupby('age').count())
print(data.isnull().sum())
# print(data.loc[data['age']== 18].sample(n= 50, replace= False))

# trainset, valset= train_test_split(data, test_size= 0.2, stratify= data['age'], random_state= 42)

# for age in ages:
#     print(age)
#     print(len(data.loc[data['age']== age]) )
#     train, val= train_test_split(data.loc[data['age']== age], test_size= 50/len(data.loc[data['age']== age]), random_state= 42)
#     trainset= pd.concat([trainset, train])
#     valset= pd.concat([valset, val])
#     print(trainset.shape, valset.shape)

# print(trainset.groupby('age').count())
# print(valset.groupby('age').count())


print(trainset.groupby('age').count())
print(valset.groupby('age').count())



trainset.to_csv('/opt/ml/code2/trainset.csv')
valset.to_csv('/opt/ml/code2/valset.csv')


# data= pd.read_csv('/opt/ml/code2/trainset.csv')
# data= data[data['age']>=57]
# print(len(data[data['age']>=57]))
# data.to_csv('/opt/ml/code2/oldset_57.csv')
# print(data)
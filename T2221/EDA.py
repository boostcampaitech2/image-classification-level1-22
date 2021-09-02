import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# if __name__ == '__main__':
#     data= pd.read_csv('/opt/ml/code2/submission_ff.csv')
#     print(data.groupby('ans').count())
#     print(len(data.groupby('ans').count().index), len(data.groupby('ans').count()))

#     plt.bar(data.groupby('ans').count().index, data.groupby('ans').count()['ImageID'])
#     # # sns.displot(data, x= 'ans', stat='density')
#     plt.savefig('/opt/ml/code2/EDA_sub_fin.png')


if __name__ == '__main__':
    num= 6
    data_org= pd.read_csv(f'/opt/ml/code2/ensemble/final/submission_test_final.csv')
    plt.bar(data_org.groupby('ans').count().index, data_org.groupby('ans').count()['ImageID'])

    plt.savefig(f'/opt/ml/code2/ensemble/final/EDA_fin.png')
    # for i in range(3):
    #     fig = plt.figure()
    #     name= f'/opt/ml/code2/val_pred{i}'
    #     data= pd.read_csv(name+'.csv')
    #     print(data.groupby('label').count())
    #     print(len(data.groupby('label').count().index), len(data.groupby('age').count()))

    #     plt.bar(data.groupby('label').count().index, data.groupby('label').count()['path'])
    #     plt.bar(data_org.groupby('label').count().index, data_org.groupby('label').count()['path'])

    #     plt.show()
    #     # # sns.displot(data, x= 'ans', stat='density')
    #     plt.savefig(name+'png')

# 1130 / 833 / 286
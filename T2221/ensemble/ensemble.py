import pandas as pd
from tqdm import tqdm
import numpy as np
import re

data0= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2221_age.csv')
data1= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2221_label.csv')
data2= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2046.csv')
data3= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2037.csv')
data4= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2039.csv')
data5= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2021.csv')
data6= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2101.csv')
data7= pd.read_csv('/opt/ml/code2/ensemble/csv/soft_submission_T2247.csv')

# print(data1['ans'][0][2])
data= re.split('[\[\]\s]+', data1['ans'][0])

submission_final= pd.read_csv('/opt/ml/input/data/eval/info.csv')

# print(data[1:-1])
# print(data2['ans'][0])
# print(data1['ans'][0]+ data2['ans'][0])

predictions= []
for i in tqdm(range(len(data1))):
    tmp0= np.array(list(map(float,re.split('[\[\]\s]+', data0['ans'][i])[1:-1])))
    tmp1= np.array(list(map(float,re.split('[\[\]\s]+', data1['ans'][i])[1:-1])))
    tmp2= np.array(list(map(float, re.split('[\[\]\s]+', data2['ans'][i])[1:-1])))
    tmp3= np.array(list(map(float, re.split('[\[\]\s]+', data3['ans'][i])[1:-1])))
    tmp4= np.array(list(map(float, re.split('[\[\]\s]+', data4['ans'][i])[1:-1])))
    tmp5= np.array(list(map(float, re.split('[\[\]\s]+', data5['ans'][i])[1:-1])))
    tmp6= np.array(list(map(float, re.split('[\[\]\s]+', data6['ans'][i])[1:-1])))
    tmp7= np.array(list(map(float, re.split('[\[\]\s]+', data7['ans'][i])[1:-1])))



    tmp= tmp0+ tmp1 + tmp2 + tmp3 + tmp4+ tmp5+ tmp6+ tmp7
    tmp_idx= np.argmax(tmp)
    # print(len(tmp1), len(tmp2))
    # for j in range(18):
    #     tmp[j]= tmp1[j]+ tmp2[j]

    predictions.append(tmp_idx)

submission_final['ans']= predictions

submission_final.to_csv('/opt/ml/code2/ensemble/final/submission_test_final.csv')
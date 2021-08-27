import os
import pandas as pd
import matplotlib.pyplot as plt
import termplotlib as tpl
import numpy as np

train_path = '../input/data/train'
train_img = '../input/data/train/images'

test_path = '../input/data/eval'

train_csv = pd.read_csv(os.path.join(train_path, 'train_modified.csv'))

label_count = train_csv["label"].value_counts()

print(len(label_count))

label_max = label_count.max()

beta = 0.99

eff_num = (1.0-np.power(beta, label_count))
weights = (1.0-beta) / eff_num
weight = weights /np.sum(weights) * 18
print(eff_num)
print(weights)
print(weight)

print(np.sum(weights))
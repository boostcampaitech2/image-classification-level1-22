import pandas as pd
import os

#test_dir = '/opt/ml/input/data/eval'
test_dir = '/opt/ml/input/data/eval'
subm_append = pd.read_csv(os.path.join(test_dir,'info.csv'))

print(subm_append)

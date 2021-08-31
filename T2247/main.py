import dataset_
import train
import pandas as pd
from torch import nn
from torch import optim
CLASSES = 18

if __name__ == '__main__':
    df = pd.read_csv('/opt/ml/input/data/train/label.csv')

    train.train(df)

import torch
import numpy as np
from datetime import datetime
import pytz

class EarlyStopping:
    #https://github.com/Bjarten/early-stopping-pytorch
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_local_time(date=True):
    tz = pytz.timezone('Asia/Seoul') 
    dt = datetime.now(tz)
    if date:
        return dt.strftime("%D %H:%M:%S")
    else:
        return dt.strftime("%H:%M:%S")

def get_loss_weight(csv_file, weight_schemes = "ENS", beta = 0.99):
    label_count = csv_file["label"].value_counts()
    num_classes = len(label_count)
    
    if weight_schemes == "INS":
        loss_weight = 1 / label_count

    elif weight_schemes == "ISNS":
        loss_weight = 1 / label_count
    
    elif weight_schemes == "ENS":
        eff_num = (1.0-np.power(beta, label_count))
        weights = (1.0-beta) / eff_num
        loss_weight = weights /np.sum(weights) * num_classes

    elif weight_schemes == "MAX":
        label_max = label_count.max()
        loss_weight = label_max / label_count

    loss_weight = loss_weight.sort_index()
    loss_weight = torch.tensor(loss_weight.values).float()

    return loss_weight


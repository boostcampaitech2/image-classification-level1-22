from pytz import timezone
from datetime import datetime as dt
import numpy as np
import torch

def get_now_str():
    return dt.now().astimezone(timezone("Asia/Seoul")).strftime('%Y%m%d%H%M%S')

def get_label_count(dataset, num_classes):
    labels = torch.zeros(num_classes, dtype=torch.long)

    for _, target in dataset:
        labels[target] += 1

    return labels

def get_loss_weight(class_list, weight_schemes = "ENS", beta = 0.99, num_classes=18):
    label_count = class_list
    if torch.is_tensor(label_count): label_count = label_count.tolist()

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

    loss_weight = torch.tensor(loss_weight).float()

    return loss_weight
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytz import timezone
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data import make_data
from data.dataset import CustomDataset
from util.transform import get_transfrom
from model.custom_model import CustomResNet18, CustomEfficientNet, CustomVit
from util.slack_noti import SlackNoti
from util.custom_loss import F1_Loss
import wandb

MODEL_PATH = '/opt/ml/code/level1-22/T2039/model/saved'
INPUT_DATA_ROOT_PATH = '/opt/ml/input/data'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEED = 2021
NUM_CLASSES = 18
VAL_SPLIT = 0
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCH = 10
EARLY_STOP = 9999
NUM_ACCUM = 1
AGE_CAT_TYPE = 1 # 1: 60세 이상을 60세 이상으로 분류, 2: 58,59세도 60세 이상으로 분류

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def split_folders(df_train, test_size):
    folders = df_train['path'].unique()
    if test_size == 0:
        return folders.tolist(), list()
    else:
        train_folders, val_folders =train_test_split(folders, test_size=test_size, random_state=SEED)
        return train_folders.tolist(), val_folders.tolist()

def get_paths_labels(df_train, folders):
    img_dir = os.path.join(INPUT_DATA_ROOT_PATH, 'train/images')
    sub_df = df_train.loc[df_train['path'].isin(folders), ['path', 'file', 'class']]
    img_paths = (sub_df['path'] + '/' + sub_df['file']).tolist()
    img_paths = [os.path.join(img_dir, path) for path in img_paths]
    labels = sub_df['class'].tolist()
    return img_paths, labels

def main():
    seed_everything(SEED)

    noti = SlackNoti()

    # wandb setting
    config = {
        'epochs': NUM_EPOCH, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'val_split': VAL_SPLIT, 'early_stop': EARLY_STOP, 'gradient_accum': NUM_ACCUM
        }
    wandb.init(project='boostcamp-image-classification', entity='zgotter', config=config)
    wandb.run.name = input('wandb experiment name: ')

    # load data
    df_train = make_data.TrainData(age_cat_type=AGE_CAT_TYPE).get_data()

    # split
    train_folders, val_folders = split_folders(df_train, VAL_SPLIT)

    # paths, labels
    train_paths, train_labels = get_paths_labels(df_train, train_folders)
    val_paths, val_labels = get_paths_labels(df_train, val_folders)

    # transform
    transform = get_transfrom()

    # datasets
    datasets = {
        'train': CustomDataset(train_paths, transform, train_labels),
        'val': CustomDataset(val_paths, transform, val_labels)
    }

    # dataloaders
    datalaoders = {
        k: DataLoader(datasets[k], batch_size=BATCH_SIZE, shuffle=True, drop_last=True) for k in datasets.keys() if len(datasets[k]) > 0
    }

    # model
    #model = CustomResNet18(NUM_CLASSES)
    #model = CustomVit(NUM_CLASSES)
    model = CustomEfficientNet(NUM_CLASSES)
    model = model.to(DEVICE)
    wandb.config.update({'model': model.name})

    # Weighted Loss
    #criterion = F1_Loss(NUM_CLASSES).to(DEVICE)    
    weight_age_over_60 = torch.tensor([1., 1., 5.] * 6).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_age_over_60)

    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    best_f1 = 0.
    best_epoch = 0
    early_stop_cnt = 0

    phases = list(datalaoders.keys())

    wandb.watch(model)
    msg = f"{model.name} train start!!!"
    print(msg)
    noti.send_message(msg)

    optimizer.zero_grad() # gradient accumulation

    for epoch in range(1, NUM_EPOCH+1):
        is_early_stop = False
        for phase in phases:
            running_loss = 0
            running_acc = 0
            running_f1 = 0
            
            if phase == 'train': model.train()
            elif phase == 'val': model.eval()

            n_iter = 0
            for i, (images, labels) in enumerate(tqdm(datalaoders[phase])):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                #optimizer.zero_grad() # gradient accumulation

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(images)
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                    if phase == 'train':
                        loss.backward()

                        if i % NUM_ACCUM == 0: # gradient accumulation
                            optimizer.step()
                            optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)
                running_acc += torch.sum(preds == labels.data)
                running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                n_iter += 1

            epoch_loss = running_loss / len(datalaoders[phase].dataset)
            epoch_acc = running_acc / len(datalaoders[phase].dataset)
            epoch_f1 = running_f1 / n_iter

            msg = f"{phase} | epoch {epoch:03d}/{NUM_EPOCH:03d}, loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, f1: {epoch_f1:.4f}"
            print(msg)
            noti.send_message(msg)
            wandb.log({
                f'{phase} loss': epoch_loss,
                f'{phase} acc': epoch_acc,
                f'{phase} f1': epoch_f1,
            })

            # save checkpoint
            if (len(phases) == 2 and phase == 'val') or (len(phases) == 1) :
                if best_f1 < epoch_f1:
                    early_stop_cnt = 0
                    best_epoch = epoch
                    best_f1 = epoch_f1

                    time_str = dt.now().astimezone(timezone("Asia/Seoul")).strftime('%Y%m%d%H%M%S')
                    save_name = f"{model.name}_checkpoint_{time_str}_{epoch:03d}.pt"
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'f1_score': best_f1
                    }, os.path.join(MODEL_PATH, save_name))
                else:
                    early_stop_cnt += 1

            if early_stop_cnt == EARLY_STOP:
                is_early_stop = True
                break
        
        if is_early_stop:
            msg = f"early stopped"
            print(msg)
            noti.send_message(msg)
            break
            
    msg = f"best f1 is {best_f1:.4f} in epoch {best_epoch:03d}"
    print(msg)
    noti.send_message(msg)    


if __name__ == '__main__':
    main()
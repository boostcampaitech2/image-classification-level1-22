import argparse
import time
import warnings
import wandb

import pandas as pd
import numpy as np
import torch
import random
import os
from tqdm import tqdm

import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from model.model import Model
from config import Config

from metric.custom_loss import CustomLoss
from torchsampler import ImbalancedDatasetSampler
from data.dataset import TrainDataset, TestDataset

# torch.cuda.empty_cache()

def set_seed(SEED): # random seed를 고정해줌
    # np.random.seed(SEED)
    # random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config(): # python local에서 받아올 
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)

    args = parser.parse_args() # arg로 받은 것을 config에 담아서 return
    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.learning_rate,
        BETA= 1.0
    )

    return config

def get_data_loader(train_path, val_path,test_path, config):

    mean = (0.56, 0.52, 0.50)
    std = (0.23, 0.24, 0.24)

    train_transform= transforms.Compose([
        transforms.CenterCrop((350, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
    
    test_transform= transforms.Compose([
        transforms.CenterCrop((350, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset= TrainDataset(train_path, transforms= train_transform)
    valset= TrainDataset(val_path, transforms= test_transform)
    testset= TestDataset(test_path, base_path= '/opt/ml/input/data/eval', transforms= test_transform)

    train_loader= DataLoader(trainset, batch_size= config.BATCH_SIZE, sampler= ImbalancedDatasetSampler(trainset), num_workers= 2)
    val_loader= DataLoader(valset, batch_size= config.BATCH_SIZE, shuffle= True, num_workers= 1)
    test_loader= DataLoader(testset, batch_size= config.BATCH_SIZE, shuffle= False, num_workers= 2)
    
    return train_loader, val_loader, test_loader


def train(epoch, optim, loss_func, train_loader, val_loader, model, schedular= None, i= 0):
    
    running_loss = 0
    running_acc = 0
    running_f1= 0

    pbar= tqdm(enumerate(train_loader), total= len(train_loader))
    for idx, (images, labels) in pbar:
        model.train()
        images= images.to(device)
        labels= labels.to(device).long() # label은 long type으로 설정

        optim.zero_grad()

        if config.BETA > 0 and np.random.random()>0.5: # cutmix 작동될 확률      
            lam= 0.5
            indices = torch.randperm(images.size(0))
            patch_images= images.clone()
            patch_labels= labels.clone()
            patch_labels= patch_labels[indices]
            W= images.size()[3]

            images_cut= images.clone()
            images_cut[:, :, :, :W//2] = patch_images[indices, :, :, :W//2]
            outputs = model(images_cut)
            
            loss = loss_func(outputs, labels) * lam + loss_func(outputs, patch_labels) * lam

        else:
            outputs= model(images) # 예측값
            loss= loss_func(outputs, labels) # loss 측정 여기서 중요한 것이 labels shape이 1차원

        _, preds= torch.max(outputs, 1) # outputs 중 max값을 1로 변경시켜줌 

        loss.backward()
        optim.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        running_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')


    model.eval()
    with torch.no_grad():
        val_loss, val_acc, val_f1= valid_model(model, val_loader, loss_func, i)

    train_loss=  running_loss/ len(train_loader.dataset)
    train_acc=  running_acc/ len(train_loader.dataset)
    train_f1= running_f1/ len(train_loader)

    print(f'\nEpoch: {epoch} train loss: {train_loss}, train acc: {train_acc}, train f1: {train_f1}\
            val loss: {val_loss}, val acc: {val_acc}, val_f1: {val_f1}')
    wandb.log({'train_acc': train_acc, 'train_f1': train_f1, 'val_acc': val_acc,'val_f1': val_f1 ,
        'val loss': val_loss, 'val acc': val_acc, 'val_f1': val_f1})
    
    if schedular is not None:
        schedular.step()
    


def valid_model(model, val_loader, loss_func, i):
    val_acc= 0
    val_loss= 0
    val_f1= 0

    pbar= tqdm(enumerate(val_loader), total= len(val_loader))
    for idx, (images, labels) in pbar:
        images= images.to(device)
        labels= labels.to(device)

        outputs= model(images)

        _, preds= torch.max(outputs, 1) # outputs 중 max값을 1로 변경시켜줌 
        loss= loss_func(outputs, labels)

        val_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
        val_acc += torch.sum(preds == labels.data)
        val_loss += loss
    i+=1

    val_loss_=  val_loss/ len(val_loader.dataset)
    val_acc_= val_acc/ len(val_loader.dataset)
    val_f1_= val_f1/ len(val_loader)

    return val_loss_, val_acc_, val_f1_

def make_submission(model, test_loader, submission, sub_path):

    model.eval()
    all_predictions = []
    for images in tqdm(test_loader):
        with torch.no_grad():
            images = images.to(device)
            preds = model(images)
            pred = preds.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            soft_predictions.extend(preds.cpu().numpy())


    submission['ans'] = all_predictions

    submission.to_csv(sub_path, index=False)
    
    submission['ans']= soft_predictions
    submission.to_csv(f'/opt/ml/code2/submission/soft_submission.csv', index=False)

if __name__=='__main__':

    
    set_seed(42)
    config= get_config()

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device) 
    print(config)
    
    IMG_WIDTH, IMG_HEIGHT= 384, 512
    TRAIN_CSV_PATH= './trainset_LABEL.csv'
    VAL_PATH= './valset_LABEL.csv'
    TEST_PATH= '/opt/ml/input/data/eval/info.csv'
    submission=  pd.read_csv(TEST_PATH)

    model= Model(18)
    # model = torch.load('/opt/ml/code/model_pt/eff_model_0.719.pt')
    model.to(device)  

    wandb.init(project= 'img_clf1')
    wandb.config.update(config)
    wandb.watch(model)

    # loss_func= torch.nn.CrossEntropyLoss()
    loss_func= CustomLoss(18)
    optm= torch.optim.Adam(model.parameters(), lr= config.LEARNING_RATE)
    schedular= torch.optim.lr_scheduler.CyclicLR(optimizer=optm, base_lr= 1e-7,\
                                            max_lr= config.LEARNING_RATE, 
                                            step_size_up= 2, step_size_down= 4, 
                                            cycle_momentum= False, verbose= True)

    train_loader, val_loader, test_loader= get_data_loader(TRAIN_CSV_PATH, VAL_PATH, TEST_PATH, config)

    i= 0
    print('\ntrain start')
    for epoch in range(config.EPOCHS):
        train(epoch, optm, loss_func, train_loader, val_loader, model, schedular, i)
    print('\ntrain fin')

    make_submission(model, test_loader, submission, f'/opt/ml/code2/submission/CUT_F1CE_IMB_DATA_STRATIFY_EPOCH{epoch}.csv')
    torch.save(model, '/opt/ml/code2/model/CUT_F1CE_IMB_DATA_STRATIFY_EPOCH{epoch}')



import argparse
import time
import warnings
import wandb

import pandas as pd
import numpy as np
import torch
import random
import os
import tqdm

import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from dataset import Image_Dataset
from model import Model
from config import Config

from custom_loss import CustomLoss
from torchsampler import ImbalancedDatasetSampler
from focal_loss import FocalLoss
from catalyst.data.sampler import BalanceClassSampler

# torch.cuda.empty_cache()

def set_seed(SEED): # random seed를 고정해줌
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_config(): # python local에서 받아올 
    parser = argparse.ArgumentParser(description='hyperparameter')
    parser.add_argument('--epochs', default=30, type=int)
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

def get_data_loader(train_path,test_path, config):

    train= pd.read_csv(train_path)
    test= pd.read_csv(os.path.join(test_path, 'info.csv'))
    test_img_dir= os.path.join(TEST_PATH, 'images')
    image_paths = [os.path.join(test_img_dir, img_id) for img_id in test.ImageID]

    mean = (0.56, 0.52, 0.50)
    std = (0.23, 0.24, 0.24)

    train_trans= transforms.Compose([
        transforms.CenterCrop((350, 300)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
        ])
    
    test_trans= transforms.Compose([
        transforms.CenterCrop((350, 300)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    
    trainset, valset= train_test_split(train, test_size= 0.2, random_state= 42)

    trainset_= Image_Dataset(trainset, transform= train_trans)
    valset_= Image_Dataset(valset, transform= test_trans)
    testset_= Image_Dataset(image_paths, transform= test_trans, train= False)

    train_loader= DataLoader(trainset_, batch_size= config.BATCH_SIZE, shuffle= True, num_workers= 1)
    val_loader= DataLoader(valset_, batch_size= config.BATCH_SIZE, shuffle= True, num_workers= 1)
    test_loader= DataLoader(testset_, batch_size= 1, shuffle= False, num_workers= 3)
    
    return train_loader, val_loader, test_loader

def train(epoch, optim, loss_func, train_loader, val_loader, model, schedular= None):
    
    running_loss = 0
    running_acc = 0
    running_f1= 0
    n_iter= 0
    pbar= tqdm.tqdm(enumerate(train_loader), total= len(train_loader), position= True, leave= True)

    for idx, (images, labels) in pbar:
        model.train()
        images= images.to(device)
        labels= labels.to(device).long() # label은 long type으로 설정

        optim.zero_grad()

        if config.BETA > 0 and np.random.random()>0.5: # cutmix 작동될 확률      
                lam = np.random.beta(config.BETA, config.BETA)
                rand_index = torch.randperm(images.size()[0]).to(device)
                target_a = labels
                target_b = labels[rand_index]            
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                outputs = model(images)
                loss = loss_func(outputs, target_a) * lam + loss_func(outputs, target_b) * (1. - lam)

        else:
            outputs= model(images) # 예측값
            loss= loss_func(outputs, labels) # loss 측정 여기서 중요한 것이 labels shape이 1차원

        _, preds= torch.max(outputs, 1) # outputs 중 max값을 1로 변경시켜줌 

        loss.backward()
        optim.step()

        running_loss += loss.item() * images.size(0)
        running_acc += torch.sum(preds == labels.data)
        running_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
        n_iter+=1
    # print('\npreds', preds,'\nlabels:', labels)


    model.eval()
    with torch.no_grad():
        val_loss, val_acc, val_f1= valid_model(model, val_loader, loss_func)

    train_loss=  running_loss/ len(train_loader.dataset)
    train_acc=  running_acc/ len(train_loader.dataset)
    train_f1= running_f1/ n_iter

    print(f'\nEpoch: {epoch} train loss: {train_loss}, train acc: {train_acc}, train f1: {train_f1}\
            val loss: {val_loss}, val acc: {val_acc}, val_f1: {val_f1}')
    wandb.log({'train_acc': train_acc, 'train_f1': train_f1, 'val_acc': val_acc,'val_f1': val_f1})
    
    if schedular is not None:
        schedular.step()


def valid_model(model, val_loader, loss_func):
    val_acc= 0
    val_loss= 0
    val_f1= 0

    n= 0
    for idx, (images, labels) in enumerate(val_loader):
        images= images.to(device)
        labels= labels.to(device)

        outputs= model(images)

        _, preds= torch.max(outputs, 1) # outputs 중 max값을 1로 변경시켜줌 
        loss= loss_func(outputs, labels)

        val_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
        val_acc += torch.sum(preds == labels.data)
        val_loss += loss
        n+=1
    val_loss_=  val_loss/ len(val_loader.dataset)
    val_acc_= val_acc/ len(val_loader.dataset)
    val_f1_= val_f1/ n

    return val_loss_, val_acc_, val_f1_

def make_submission(model, test_loader, submission, sub_path):

    model.eval()
    all_predictions = []
    for images in tqdm.tqdm(test_loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

    submission['ans'] = all_predictions

    submission.to_csv(sub_path, index=False)

def save_image(data_loader, signal):
    if signal== 'train':
        X, y= next(iter(data_loader))
        plt.imshow(X[0].permute(1, 2, 0).cpu())
        plt.savefig('crop.png')    
    else:
        X= next(iter(data_loader))
        plt.imshow(X[0].permute(1, 2, 0).cpu())
        plt.savefig('crop.png')    

def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = W  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    # print('cx cy', cx, cy)
    # print('hi', bbx1, bbx2, bby1, bby2)
    return bbx1, bby1, bbx2, bby2

def cut_mix_img(train_data_loader):
    X, y= next(iter(train_data_loader))
    lam= np.random.beta(1.0, 1.0)
    rand_index= torch.randperm(X.size()[0]).to(device)
    # print(f'rand_idx {rand_index}, {rand_index.shape}')
    shuffled_y= y[rand_index]

    # print(lam)
    # print(rand_index)
    bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
    X[:,:,bbx1:bbx2, bby1:bby2] = X[rand_index,:,bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
    print(lam)

    plt.imshow(X[0].permute(1, 2, 0).cpu())
    plt.savefig('/opt/ml/code/img/cutmix.png')    


if __name__=='__main__':

    wandb.init(project= 'img_clf1')
    
    set_seed(42)
    config= get_config()

    wandb.config.update(config)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device) 
    print(config)
    
    IMG_WIDTH, IMG_HEIGHT= 384, 512
    TRAIN_CSV_PATH= '/opt/ml/code/train_with_label_57.csv'
    TEST_PATH= '/opt/ml/input/data/eval'
    submission=  pd.read_csv(os.path.join(TEST_PATH, 'info.csv'))

    SIZE= [config.BATCH_SIZE, 3, IMG_WIDTH, IMG_HEIGHT]

    model= Model()
    # model = torch.load('/opt/ml/code/model_pt/eff_model_0.719.pt')
    model.to(device)  

    wandb.watch(model)

    # loss_func= torch.nn.CrossEntropyLoss()
    loss_func= CustomLoss(18)
    optm= torch.optim.Adam(model.parameters(), lr= config.LEARNING_RATE)
    schedular= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optm, T_max=10, eta_min=1e-6)

    train_loader, val_loader, test_loader= get_data_loader(TRAIN_CSV_PATH, TEST_PATH, config)

    # cut_mix_img(train_loader)

    print('\ntrain start')
    for epoch in range(config.EPOCHS):
        train(epoch, optm, loss_func, train_loader, val_loader, model, schedular)
    print('\ntrain fin')

    make_submission(model, test_loader, submission,'/opt/ml/code/submission/submission_f1CE_57_CUT6.csv')

    torch.save(model, '/opt/ml/code/model_pt/eff_f1CE_55_CUT6.pt')

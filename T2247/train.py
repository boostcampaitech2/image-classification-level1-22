from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader
import dataloader_
import model
import torch
import tqdm
import time
from torch import nn
from torch import optim
import pandas as pd
import copy
import os
import wandb
import dataset_
from sklearn.metrics import f1_score


N_SPLITS = 5
BATCH_SIZE = 32
NUM_WORKERS = 3
DEVICE = torch.device('cuda')
EPOCHS = 8
CLASSES = 18
LEARNING_RATE = 0.001

def loss_batch(loss_func, x, y, output, opt=None):
    loss = loss_func(output, y)
    metric_b = metrics_batch(y, output)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), metric_b

def metrics_batch(target, output):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_epoch(model_, loss_func, d_loader, opt=None):
    loss = 0.0
    metric = 0.0
    len_data = len(d_loader.dataset)

    for xb, yb in d_loader:
        xb = xb.type(torch.float).to(DEVICE)
        yb = yb.to(DEVICE)
        yb_h = model_(xb)

        loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
        loss += loss_b
        if metric_b is not None:
            metric += metric_b
    loss /= len_data
    metric /= len_data
    return loss, metric

def createDir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('OSerror')


if __name__ == '__main__':
    path_ = '/opt/ml/T2247/'
    test_dir = '/opt/ml/input/data/eval'
    wandb.init(project='darknet53', entity='kkami')

    config = wandb.config
    config.learning_rate = LEARNING_RATE

    #createDir('/opt/ml/T2247/models')
    best_loss = float('inf')
    df = pd.read_csv('/opt/ml/input/data/train/label.csv')
    skf = StratifiedKFold(n_splits=N_SPLITS)
    start_time = time.time()
    model_ = model.mynet(CLASSES).to(DEVICE)
    for i, (train_idx, val_idx) in enumerate(skf.split(df['fname'],df['label'])):
        if i != 0:
            model_.load_state_dict(best_model_wts)
             #torch.load(model_.state_dict(), path_ + pth_name)
        print(f'-------Fold : {i+1}---------')
        train_loader, val_loader = dataloader_.getDataloader(train_idx, val_idx,BATCH_SIZE,NUM_WORKERS)
 

        loss_func = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model_.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            model_.train()
            train_loss, train_metric = loss_epoch(model_, loss_func, train_loader,opt)
            model_.eval()
            wandb.log({'train_loss': train_loss})
            wandb.log({'train_metric': train_metric})
            
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model_, loss_func, val_loader)
            wandb.log({'val_loss': val_loss})
            wandb.log({'val_metric': val_metric})
            accuracy = 100 * val_metric
            wandb.log({'accuracy':accuracy})
            print(f'epoch: {epoch}, train loss: {train_loss:0.6f}, val loss: {val_loss:0.6f}, accuracy: {accuracy:0.2f}, time: {(time.time()-start_time)/60} min')

            if val_loss < best_loss:
                print(val_loss)
                bi = i
                bepoch = epoch
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model_.state_dict())
                pth_name = f'Fold{bi}_epoch{bepoch}_.pt'
                #os.path.join
                print(path_ + pth_name)
                torch.save(model_.state_dict(), path_ + pth_name)
    
    subm_dataset = dataset_.Submission_Dataset(test_dir)
    subm_dl = DataLoader(subm_dataset, shuffle=False)
    subm_append = pd.read_csv(os.path.join(test_dir,'info.csv'))

    all_prediction = []
    for imgs in subm_dl:
        with torch.no_grad():
            imgs = imgs.to(DEVICE)
            pred = model_(imgs)
            pred = pred.argmax(dim=-1)
            #_, f1pred = torch.max(pred,1)
            #f1_ = f1_score(f1pred.cpu().numpy(), la)
            all_prediction.extend(pred.cpu().numpy())
    subm_append['ans'] = all_prediction

    subm_append.to_csv(os.path.join(test_dir, 'test_darknet53_subm01.csv'), index=False)




    

import argparse
from parse_config import json_to_config
from importlib import import_module
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from metric.custom_loss import create_criterion
from util.custom_util import get_now_str
from util.slack_noti import SlackNoti
import wandb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(config):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # set seed
    seed_everything(config['seed'])

    # slack noti setting
    if config['slack_noti']['use'] == 'True':
        noti = SlackNoti(config['slack_noti']['url'])

    # wandb setting
    if config['wandb']['use'] == 'True':
        wandb_config = {
            'seed': config['seed'],
            'epochs': config['num_epoch'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'val_ratio': config['val_ratio'],
            'loss': config['criterion'],
            'early_stop': config['early_stop']
        }
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=wandb_config
        )
        wandb.run.name = input('wandb experiment name: ')

    # dataset
    dataset_module = getattr(import_module('data.dataset'), config['dataset'])
    dataset = dataset_module(
        data_dir=config['path']['train_data'],
        val_ratio=config['val_ratio'],
        seed = config['seed'],
        mean=(0.560, 0.524, 0.501),
        std=(0.617, 0.587, 0.568)        
    )

    # augmentation
    transform_module = getattr(import_module('data.dataset'), config['augmentation']['train'])
    transform = transform_module(
        #resize=[128, 96],
        #mean=dataset.mean,
        #std=dataset.std
    )

    dataset.set_transform(transform)

    train_dataset, val_dataset = dataset.split_dataset()

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True
    )

    val_dataloader = None
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=True            
        )

    # model
    model_module = getattr(import_module('model.custom_model'), config['model'])
    model = model_module(dataset.num_classes).to(device)
    if config['wandb']['use'] == 'True':
        wandb.config.update({'model': model.name})

    # loss
    weight_age_over_60 = torch.tensor([1., 1., 5.] * 6).to(device)

    # 1) cross entropy loss or Focal loss
    criterion = create_criterion(config['criterion'], weight=weight_age_over_60)

    # 2) label smoothing loss
    #criterion = create_criterion(config['criterion'], classes=dataset.num_classes, smoothing=0.5)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training
    best_f1 = 0.
    best_epoch = 0
    early_stop_cnt = 0
    old_file = ''

    if config['wandb']['use'] == 'True':
        wandb.watch(model)

    for epoch in range(1, config['num_epoch']+1):

        check_f1 = 0.
        ############### train ###############
        running_loss = 0
        running_acc = 0
        running_f1 = 0

        model.train()

        for images, labels in tqdm(train_dataloader):
            images = images.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad()
            logits = model(images)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_acc += torch.sum(preds == labels.data)
            running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

        train_loss = running_loss / len(train_dataloader.dataset)
        train_acc = running_acc / len(train_dataloader.dataset)
        train_f1 = running_f1 / len(train_dataloader)
        check_f1 = train_f1

        msg = f"train | epoch {epoch:03d}/{config['num_epoch']:03d}, loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}"
        print(msg)
        if config['slack_noti']['use'] == 'True':
            noti.send_message(msg)
        if config['wandb']['use'] == 'True':
            wandb.log({
                f'train loss': train_loss,
                f'train acc': train_acc,
                f'train f1': train_f1,
            })

        ############### eval ###############
        if val_dataloader != None:
            with torch.no_grad():
                running_loss = 0
                running_acc = 0
                running_f1 = 0

                model.eval()

                for images, labels in tqdm(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device) 

                    logits = model(images)
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)

                    running_loss += loss.item() * images.size(0)
                    running_acc += torch.sum(preds == labels.data)
                    running_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

                val_loss = running_loss / len(val_dataloader.dataset)
                val_acc = running_acc / len(val_dataloader.dataset)
                val_f1 = running_f1 / len(val_dataloader)
                check_f1 = val_f1

                msg = f"val | epoch {epoch:03d}/{config['num_epoch']:03d}, loss: {val_loss:.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}"
                print(msg)
                if config['slack_noti']['use'] == 'True':
                    noti.send_message(msg)
                if config['wandb']['use'] == 'True':
                    wandb.log({
                        f'val loss': val_loss,
                        f'val acc': val_acc,
                        f'val f1': val_f1,
                    })

        ############### save checkpoint ###############
        if best_f1 < check_f1:
            early_stop_cnt = 0
            best_epoch = epoch
            best_f1 = check_f1

            remove_file_path = os.path.join(config['path']['save'], old_file)
            if os.path.isfile(remove_file_path):
                os.remove(remove_file_path)

            save_name = f"{model.name}_checkpoint_{get_now_str()}_{epoch:03d}.pt"
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'f1_score': best_f1
            }, os.path.join(config['path']['save'], save_name))  
            old_file = save_name
        else:
            early_stop_cnt += 1

        ############### early stopping ###############
        if early_stop_cnt == config['early_stop']:
            msg = f"early stopped"
            print(msg)
            if config['slack_noti']['use'] == 'True':
                noti.send_message(msg)
            break            

    msg = f"best f1 is {best_f1:.4f} in epoch {best_epoch:03d}"
    print(msg)
    if config['slack_noti']['use'] == 'True':
        noti.send_message(msg)    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Mask Wear Status Classification')
    args.add_argument('-c', '--config', default=None, type=str, help='config.json file path')
    config = json_to_config(args)
    main(config)
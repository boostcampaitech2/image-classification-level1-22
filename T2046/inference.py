import argparse
from parse_config import json_to_config
from importlib import import_module
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from util.custom_util import get_now_str
import torch
from torch.utils.data import DataLoader
from util.slack_noti import SlackNoti

def main(config):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if config['slack_noti']['use'] == 'True':
        noti = SlackNoti(config['slack_noti']['url'])

    # dataset
    dataset_module = getattr(import_module('data.dataset'), config['dataset'])
    dataset = dataset_module(
        data_dir = config['path']['eval_data'],
        val_ratio = 0,
        seed = config['seed'],
        mean=(0.532, 0.476, 0.448),
        std=(0.593, 0.544, 0.519)
    )

    # augmentation
    transform_module = getattr(import_module('data.dataset'), config['augmentation']['submission'])
    transform = transform_module(
        #resize=[128, 96],
        #mean=dataset.mean,
        #std=dataset.std
    )

    dataset.set_transform(transform)    

    # dataset & dataloader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    # model
    model_module = getattr(import_module('model.custom_model'), config['model'])
    model = model_module(dataset.num_classes)

    checkpoint = torch.load(glob(os.path.join(config['path']['save'], '*.pt'))[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    model.eval()

    preds = []
    for images in tqdm(dataloader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
    
    df_info = pd.read_csv(os.path.join(config['path']['eval_data'], 'info.csv'))
    df_info['ans'] = preds

    # save
    df_info.to_csv(os.path.join(config['path']['submission'], 
                                f"submission_{config['submission_no']}_{get_now_str()}.csv"), 
                   index=False)
                   
    msg = 'inference finished'
    print(msg)
    if config['slack_noti']['use'] == 'True':
        noti.send_message(msg)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Mask Wear Status Classification')
    args.add_argument('-c', '--config', default=None, type=str, help='config.json file path')
    config = json_to_config(args)
    main(config)
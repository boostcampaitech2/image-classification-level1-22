import pandas as pd
import os
from tqdm import tqdm
from pytz import timezone
from datetime import datetime as dt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from data.dataset import CustomDataset
from util.transform import get_transfrom
from model.custom_model import CustomResNet18, CustomEfficientNet, CustomVit
from util.slack_noti import SlackNoti

MODEL_PATH = '/opt/ml/code/github/model/saved'
TEST_PATH = '/opt/ml/input/data/eval'
SUBMISSION_PATH = '/opt/ml/code/github/submission'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 18

def main():
    noti = SlackNoti()

    df_info = pd.read_csv(os.path.join(TEST_PATH, 'info.csv'))
    IMG_PATH = os.path.join(TEST_PATH, 'images')
    img_paths = [os.path.join(IMG_PATH, img_id) for img_id in df_info.ImageID]
    
    # transform
    transform = get_transfrom()

    # dataset & dataloader
    dataset = CustomDataset(img_paths, transform)
    dataloader = DataLoader(dataset, shuffle=False)
    
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'efficientNet_checkpoint_20210826185154_007.pt'))
    #model = CustomResNet18(NUM_CLASSES)
    #model = CustomVit(NUM_CLASSES)
    model = CustomEfficientNet(NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    model.eval()

    preds = []
    for images in tqdm(dataloader):
        with torch.no_grad():
            images = images.to(DEVICE)
            pred = model(images)
            pred = pred.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
    
    df_info['ans'] = preds

    # save
    time_str = dt.now().astimezone(timezone("Asia/Seoul")).strftime('%Y%m%d%H%M%S')
    df_info.to_csv(os.path.join(SUBMISSION_PATH, f"submission_18_{time_str}.csv"), index=False)
    msg = 'inference finished'
    print(msg)
    noti.send_message(msg)

if __name__ == '__main__':
    main()
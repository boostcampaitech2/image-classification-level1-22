import os
import pandas as pd
from torch.utils.data import DataLoader

from model import *
from data.dataset import *
from utils import *

from importlib import import_module

import tqdm

import argparse
from torchvision import transforms

def submission(args, transform):
    submission = pd.read_csv(os.path.join(args.test_dir, 'info.csv'))
    image_dir = os.path.join(args.test_dir, 'images')

    device = torch.device('cuda')

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=18).to(device)
    if args.model=='EfficientNetMSD':
        model = model_module(
            num_classes = 18,
            sample_size = args.sample_size
        ).to(device)
    
    model.load_state_dict(torch.load(args.model_path))

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )
    
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    if args.submission_dir != '':
        submission_path = os.path.join(args.submission_dir, args.submission_file)
    else:
        submission_path = os.path.join(args.test_dir, args.submission_file)
    
    submission.to_csv(submission_path, index=False)
    print('test inference is done!')
    print(f'submission file saved at {submission_path}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='EfficientNetModel', help='model type (default: EfficientNetModel)')
    parser.add_argument('--model_path', type=str, default='', help='model checkpoint path')
    parser.add_argument('--submission_file', type=str, default='submission.csv', help='submission file name')
    parser.add_argument('--submission_dir', type=str, default='', help='path to save submission file')
    parser.add_argument('--test_dir', type=str, default='/opt/ml/input/data/eval', help='path of evaluation data')
    
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.CenterCrop(350),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("Start Testing...")
    submission(args, transform)
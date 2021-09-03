import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import ToTensor

from dataset import TestDataset

def get_f1_score(y_pred, y_true):
    num_classes = 18
    epsilon = 1e-7

    y_pred = F.softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred, 1)

    y_pred = F.one_hot(y_pred, num_classes).to(torch.float32)
    y_true = F.one_hot(y_true, num_classes).to(torch.float32)
    
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1.mean()
        
def main():
    test_dir = '/opt/ml/input/data/eval'

    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        ToTensor(),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False,
    )

    device = torch.device('cuda')
    model = torch.load('best.pt')
    model.eval()

    all_predictions = []
    soft_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            soft_pred = F.softmax(pred, dim=1)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            soft_predictions.extend(soft_pred.cpu().numpy())
    submission['ans'] = soft_predictions
    
    submission.to_csv('soft_submission_T2101.csv', index=False)
    print('test inference is done!')

if __name__ == '__main__':
    main()
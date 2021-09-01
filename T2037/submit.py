import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop

from model import EfficientNet
import tqdm
import albumentations as A
import cv2

torch.manual_seed(42)
test_dir = "/opt/ml/input/data/eval"


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, "info.csv"))
image_dir = os.path.join(test_dir, "images")

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]


transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor(),
    Normalize(
        mean=(0.4914, 0.4822, 0.4465), 
        std=(0.2023, 0.1994, 0.2010)
        )
    ])


dataset = TestDataset(image_paths, transform)

loader = DataLoader(dataset, shuffle=False)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device("cuda")

model = EfficientNet(18)
model.load_state_dict(
    torch.load(
        "/opt/ml/code/image-classification-level1-22/T2037/model/Album_crossentropy_facecrop_labelfix2/best.pth"
    )
)
model.to(device)
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []

pbar = tqdm.tqdm(loader, total=len(loader), position=True, leave=True)

for images in pbar:
    with torch.no_grad():
        images = images.to(device)
        preds = model(images)
        preds = preds.argmax(dim=1)
        all_predictions.extend(preds.cpu().numpy())
submission["ans"] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(
    os.path.join(test_dir, "EfficientNet_Album_submission.csv"), index=False
)
print("test inference is done!")

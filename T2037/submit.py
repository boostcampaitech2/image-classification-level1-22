import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop

from model import ViTBase16
import tqdm

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
    CenterCrop((384, 384)),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset = TestDataset(image_paths, transform)

loader = DataLoader(dataset, shuffle=False)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device("cuda")

model_age = torch.load("/opt/ml/model_only_age/efficient_Thu Aug 26 05:08:09 2021_0.9889599084854126.pt")
model_jender = torch.load("/opt/ml/model_only_jender/efficient_Thu Aug 26 05:44:23 2021_0.9956563711166382.pt")
model_mask = torch.load("/opt/ml/model_only_mask/efficient_Thu Aug 26 06:17:19 2021_0.9958373308181763.pt")
model_age.to(device)
model_jender.to(device)
model_mask.to(device)
model_age.eval()
model_jender.eval()
model_mask.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
pbar = tqdm.tqdm(loader, total= len(loader), position= True, leave= True)
for images in pbar:
    with torch.no_grad():
        images = images.to(device)
        age = model_age(images).argmax(dim=-1)
        jender = model_jender(images).argmax(dim=-1)
        mask = model_mask(images).argmax(dim=-1)

        preds = age + (jender * 3) + (mask*6)
        all_predictions.extend(preds.cpu().numpy())
submission["ans"] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, "submission.csv"), index=False)
print("test inference is done!")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
import tqdm

device = torch.device("cuda")

mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = "/opt/ml/data/train/new_imgs"
img_path = "/opt/ml/data/train/images"

cnt = 0

pbar = tqdm.tqdm(
    os.listdir(img_path),
    total=len(os.listdir(img_path)),
    position=True,
    leave=True,
)

for paths in pbar:
    if paths[0] == ".":
        continue

    sub_dir = os.path.join(img_path, paths)

    for imgs in os.listdir(sub_dir):
        if imgs[0] == ".":
            continue

        img_dir = os.path.join(sub_dir, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mtcnn 적용
        boxes, probs = mtcnn.detect(img)

        # boxes 확인
        if len(probs) > 1:
            # print(boxes)
            pass
        if not isinstance(boxes, np.ndarray):
            # print("Nope!")
            # 직접 crop
            img = img[100:400, 50:350, :]

        # boexes size 확인
        else:
            xmin = int(boxes[0, 0]) - 30
            ymin = int(boxes[0, 1]) - 30
            xmax = int(boxes[0, 2]) + 30
            ymax = int(boxes[0, 3]) + 30

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > 384:
                xmax = 384
            if ymax > 512:
                ymax = 512

            img = img[ymin:ymax, xmin:xmax, :]

        tmp = os.path.join(new_img_dir, paths)
        cnt += 1

        try:
            os.makedirs(tmp)
        except:
            pass
        plt.imsave(os.path.join(tmp, imgs), img)

print(cnt)

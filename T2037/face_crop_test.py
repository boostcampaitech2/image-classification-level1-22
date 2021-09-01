import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
import tqdm
from retinaface import RetinaFace

device = torch.device("cuda")

mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = "/opt/ml/data/eval/retina_imgs"
img_path = "/opt/ml/data/eval/images"

cnt = 0

pbar = tqdm.tqdm(
    os.listdir(img_path),
    total=len(os.listdir(img_path)),
    position=True,
    leave=True,
)

for img_dir in pbar:

    if img_dir[0] == ".":  # 더미 파일 패스
        continue

    img = cv2.imread(os.path.join(img_path, img_dir))
    # print(os.path.join(img_path, img_dir))
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
        result_detected = RetinaFace.detect_faces(img_dir)
        if type(result_detected) == dict:
            # print("retina success!")
            xmin = int(result_detected["face_1"]["facial_area"][0]) - 30
            ymin = int(result_detected["face_1"]["facial_area"][1]) - 30
            xmax = int(result_detected["face_1"]["facial_area"][2]) + 30
            ymax = int(result_detected["face_1"]["facial_area"][3]) + 30
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > 384:
                xmax = 384
            if ymax > 512:
                ymax = 512

            img = img[ymin:ymax, xmin:xmax, :]
        else:
            # print("retina fail!!!!!!!!!!!")
            img = img[100:400, 50:350, :]
    else:
        # print("mtcnn success!")
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

    # boexes size 확인

    tmp = os.path.join(new_img_dir, img_dir)
    cnt += 1

    img.save(new_img_dir, img)

print(cnt)

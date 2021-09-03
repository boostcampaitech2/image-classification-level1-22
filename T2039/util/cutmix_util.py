# https://www.kaggle.com/kaushal2896/cifar-10-simple-cnn-with-cutmix-using-pytorch
# https://sseunghyuns.github.io/classification/2021/05/25/invasive-pytorch/#

#from util.custom_util import get_now_str
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def shuffle_minibatch(images, labels):
    """
    배치 순서를 섞어주는 함수
    """
    assert images.size(0) == labels.size(0)
    indices = torch.randperm(images.size(0))
    return images[indices], labels[indices]

def rand_bbox(size, lam):
    """
    random으로 cutmix 할 영역을 찾아주는 함수
    size: [B, C, W, H]
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam) # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat) # 패치의 너비
    cut_h = np.int(H * cut_rat) # 패치의 높이

    # 기존 이미지의 크기에서 랜덤하기 값을 가져온다. (중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2    

def quarter_bbox(size, lam):
    positions = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    idx = np.random.randint(0, 4)

    center_h = size[2]//2
    center_w = size[3]//2
    position = positions[idx]

    bbx1 = center_h * position[0]
    bby1 = center_w * position[1]
    bbx2 = center_h * (position[0]+1)
    bby2 = center_w * (position[1]+1)

    return bbx1, bby1, bbx2, bby2  


def save_image(image, save_dir, epoch, batch):
    """
    이미지 저장 함수
    """
    plt.imshow(image.permute(1,2,0).cpu())
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"cutmix_{epoch:03d}_{batch:03d}.png")) # './saved/image/cutmix.png'

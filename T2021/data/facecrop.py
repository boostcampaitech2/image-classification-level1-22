# https://stages.ai/competitions/74/discussion/talk/post/475

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2


def facecrop(dir, new_dir):
    device = device = torch.device('cuda')
    mtcnn = MTCNN(keep_all=True, device=device)

    cnt = 0

    for paths in os.listdir(dir):
        if paths[0] == '.': continue
        
        sub_dir = os.path.join(dir, paths)
        
        for imgs in os.listdir(sub_dir):
            if imgs[0] == '.': continue
            
            img_dir = os.path.join(sub_dir, imgs)
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            #mtcnn 적용
            boxes,probs = mtcnn.detect(img)
            
            if not isinstance(boxes, np.ndarray):
                img=img[100:400, 50:350, :]
            
            # boexes size 확인
            else:
                xmin = int(boxes[0, 0])-30
                ymin = int(boxes[0, 1])-30
                xmax = int(boxes[0, 2])+30
                ymax = int(boxes[0, 3])+30
                
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > 384: xmax = 384
                if ymax > 512: ymax = 512
                
                img = img[ymin:ymax, xmin:xmax, :]
                
            tmp = os.path.join(new_img_dir, paths)
            cnt += 1
            plt.imsave(os.path.join(tmp, imgs), img)
            
    print(cnt)

def facecrop2(dir, new_dir):
    device = device = torch.device('cuda')
    mtcnn = MTCNN(keep_all=True, device=device)

    cnt = 0
    
    for imgs in os.listdir(dir):
        if imgs[0] == '.': continue
            
        img_dir = os.path.join(dir, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
            
        if not isinstance(boxes, np.ndarray):
            img=img[100:400, 50:350, :]
            
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30
                
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
                
            img = img[ymin:ymax, xmin:xmax, :]
                
        tmp = os.path.join(new_img_dir, imgs)
        cnt += 1
        plt.imsave(tmp, img)
            
    print(cnt)

if __name__ == "__main__":
    new_img_dir = '../../input/data/train/new_images'
    img_dir = '../../input/data/train/images'
    new_eval_dir = '../../input/data/eval/new_images'
    eval_dir = '../../input/data/eval/images'

    facecrop2(eval_dir, new_eval_dir)

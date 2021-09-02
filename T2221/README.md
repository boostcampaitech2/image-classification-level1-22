### 파일별 설명



### [config.py](http://config.py)

EPOCH, BATCH_SIZE, LEARNING_RATE, BETA 등의 하이퍼 파리미터를 조정

### [model.py](http://model.py)

timm 라이브러리에서 pretrained된 efficientnet_b0 모델을 가져옴

### custom_loss.py

loss를 customizing하여 CrossEntropyLoss와 F1 score의 loss를 합쳐주는 loss를 구현

### [dataset.py](http://dataset.py)

split_age에서 가져온 인물별로 나눠진 데이터를 labeling을 시켜 파일 저장

Traindataset, Testdataset 기본 class 지정

### [train.py](http://train.py)

cut mix와 간단한 augmentation을 이용 (centorcrop, horizontalflip, normalize), cutmix를 사용할 때 데이터의 분포 중에 나이가 제일 불균형하므로 age기준으로 나누어 균등하게 추출할 수 있도록 ImbalancedDatasetSampler를 사용하였으며 cutmix는 batch안에 있는 사람들의 순서를 바꾸어 세로로 반 잘라서 augmentation을 해주었다.

trainset, validset csv path를 넣어주고 config에 있는 파라미터들을 지정해주고 실행해주면 submission 파일과 모델이 저장이 된다.

### [ensemble.py](http://ensemble.py)

각 팀원들의 submission 파일의 label을 one-hot으로 변경하여 불러와 전처리 과정을 진행하고 soft voting 방식을 이용하여 최종 결과물을 만들어 준다.

### ImbalancedDatasetSampler

아래 링크에서 가져온 라이브러리. 데이터 불균형 현상을 개선시켜줌. Dataset의 get_labels 함수를 통해 가져온 값을 기준으로 DataLoader에서 oversampling, undersampling을 하며 균등하게 뽑아줌.

``` python
git clone https://github.com/ufoym/imbalanced-dataset-sampler.git
```



https://github.com/ufoym/imbalanced-dataset-sampler
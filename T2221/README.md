## 코드 사용 GUIDE

1. data 폴더 안에 있는 **split_age.py**를 통하여 나이별로 그룹을 지정

   **group_age_stratify** : input data를 받아와 stratify하게 trainset과 valset을 나눠줌

   **group_age_balance** : data를 총 9개의 그룹으로 main문에서 나눠줬는데, 그룹 별로 50개의 데이터를 추출하여 valset을 구성해주고, trainset은 나머지로 구성해줌

   **old_data_from_train**: trainset에서 상대적으로 data가 적은 old data를 따로 뽑아 반환

2. data 폴더 안에 있는 **dataset.py**를 사용. Make Label 클래스는 split_age.py를 통해 나눠준 trainset, valset을 가져와 학습에 사용할 수 있도록 label을 포함한 형태로 변형시켜줌. init부분의 csv_path에 받아올 csv 경로, run 부분에서 csv로 저장할 파일의 경로만 지정해주면 labeling 부분에서 알아서 계산하여 반환해줌.

   TrainDataset, TestDataset 클래스가 있는데 Traindataset은 image, label 반환, Testdataset은 label을 반환해줌. 또한 Traindataset과 TestDataset에 적용하는 transform이 다르므로 각자 클래스를 지정해주었다.

3. model 폴더 안의 **model.py**에서 사용하고자 하는 모델을 지정해주면 된다. 우리가 사용하고자 하는 task의 클래스 개수를 parameter로 지정해주면 된다. timm 라이브러리에서 제공하는 모델의 경우 num_classes로 클래스 개수를 넣어주면 마지막 fc에 자동으로 형변환을 해주므로 따로 지정해줄 필요가 없다. 또한 input도 알아서 형변환을 해주므로 사용하기 편리함.

4. metric 폴더 안의 **custom_loss.py**에서 학습에 사용할 Loss를 확인한다. CustomLoss의 경우는 F1 loss와 CrossEntropyLoss를 합하여 loss를 사용하는 것을 볼 수 있음.

5. **config.py**를 확인하여 학습 시에 사용할 수 있는 하이퍼 파라미터가 무엇인지 확인 해줌.

6. **train.py**를 이용하여 학습을 진행. main문에서 dataset.py에서 저장한 train_csv_path, val_csv_path를 지정해주고 wandb를 사용하므로 사용하게 될 project를 지정해준다. loss func, optm, schedular를 어떤 것을 사용하는지 확인을 해준 다음 get_data_loader로부터 우리가 학습과 평가에 사용하게 될 train, val, test Dataloader를 받아온다. 그 뒤 학습을 진행시켜준다. 학습이 끝나면 make_submission을 통해 csv파일을 만들어준다. 제출한 파일의 경우 epoch을 돌면서 validation f1 score가 0.74 이상일 때 make_submission과 torch.save로 모델을 저장해주어 그 중 높은 score의 파일을 제출하였다.

   **set_seed** : 학습에 사용하게 될 seed를 고정해줌.

   **get_config** : 터미널에서 실행 시에 지정을 해준 옵션들을 받아와 config 형식으로 반환

   **get_data_loader** : train, val, test set에 대한 각각의 transform을 지정해주고, dataloader로 변경하여 반환을 해준다. train_loader의 경우에는 ImbalancedDatasetSampler를 가져와 사용을 해주었다. 이 sampler를 사용하기 위해서는 dataset에서 get_labels를 추가적으로 지정을 해주어야 하는데, 필자의 경우 기존에 분류해야할 'label'과 모델이 맞추기 어려워 하는 'age'를 지정해 주어 2가지 방법으로 sampler를 사용하였다. 

   **train** : data_loader에 올라가 있는 이미지를 순서만 바꾸어 cutmix를 진행해주었다. 이 때 랜덤으로 패치 사이즈를 지정해주게 되면 label을 결정하는데 불필요한 배경, 옷 등을 패치하는 경우가 있게 되므로 centorcrop을 진행해준 뒤(get_data_loader에서 했음) 세로 방향으로 반반 잘라서 사진을 구성하는 방식을 이용하였다. 이 방법을 사용할 때 다른 방법으로 현재 우리의 데이터에서 60대의 데이터가 가장 적으므로 cutmix를 할 때 60대 이상의 데이터를 가져와 넣어주는 방식을 적용해보았지만 성능 향상에 좋은 결과를 못 미쳤던 것 같음.. 너무 많이 augmentation을 해서 30-60대 보다 60대를 더 선택하는 결과를 초래했음. valid 함수에서 평가를 한 지표를 가져와 같이 출력을 해주고 있음.

   **valid_model** : 학습 진행 중인 모델의 성능을 평가해줌.

   **make_submission** : 학습된 모델을 불러와 inference / 결과물을 2개 저장하는데 기존 submission 형태 1개와 ensemble 시에 필요한 soft voting 방식으로 저장.

7. ensemble 폴더 안의 **ensemble.py** 를 통해 진행. train.py에서 make_submission을 할 때 저장된 soft submission을 불러와 soft voting 방식을 이용해 최종 결과물을 만든다. 



### ImbalancedDatasetSampler

아래 링크에서 가져온 라이브러리. 데이터 불균형 현상을 개선시켜줌. Dataset의 get_labels 함수를 통해 가져온 값을 기준으로 DataLoader에서 oversampling, undersampling을 하며 균등하게 뽑아줌.

``` python
git clone https://github.com/ufoym/imbalanced-dataset-sampler.git
```

Link : https://github.com/ufoym/imbalanced-dataset-sampler


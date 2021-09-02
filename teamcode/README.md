# 마스크 착용 상태 분류
카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task의 Solution

## Archive contents

```
├──  data
│    └── dataset.py  - file to make dataset and set augmentation
│
├──  metric
│    └── custom_loss.py  - file that contains loss functions
│
├──  model
│    └── custom_model.py  - file that contains model classes
│
├──  util
│    └── custom_util.py  - file that contains customized util functions
│    └── install package.ipynb  - file that contains customized util functions
│    └── slack_noti.py  - contains class for slack notification
│   
├──  config.json - config file for training, submission, slack and wandb
│
├── inference.py - file to make submission
│
├── parse_config.py - this file parses config.json file
│
├── train.py - file to train model
```

## Training
To train models, run following command.
```
$ python train.py -c config.json
```

## Make Submission
To make submission, run follwing command.
```
$ python inference.py -c config.json
```
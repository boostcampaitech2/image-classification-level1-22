## Face Mask Classification
My solution for Face Mask Classfication Competition
Used pretrained EfficientNet b0 and added classification layer.
Used Weighted Cross Entropy, calculated loss weight by Effective Number of Samples.

## Archive contents
```
├── checkpoint/
│   └── efficientnet-b0_best_checkpoint.pt
├── model/
│   ├── metrics.py
│   ├── loss.py
│   └── model.py
├── utils/
│   └── utils.py
├── data/
│   └── dataset.py
├── train.py
└── inference.py
```

## Training
To train model, run command.
```
$ python train.py
```

## Make Submission
To make submission file, run command.
```
$ python inference.py
```
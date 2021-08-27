import os

from torchvision import transforms

import pandas as pd

from torch.utils.data import DataLoader, random_split

from model import *
from submit import *
from data.dataset import *
from train import *

import warnings
warnings.simplefilter("ignore", UserWarning)

if __name__ == "__main__":

  # file path
  train_path = '../input/data/train'
  train_img = '../input/data/train/images'
  test_path = '../input/data/eval'
  train_csv = pd.read_csv(os.path.join(train_path, 'train_modified.csv'))

  # Get Dataset
  transform = transforms.Compose([
    transforms.CenterCrop(350),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

  mask_dataset = MaskDataset(csv_file=train_csv, img_path=train_img, transform=transform)

  train_set, val_set = random_split(mask_dataset,[17000, 1900],  generator=torch.Generator().manual_seed(42))

  batch_size = 64

  dataloaders = {
      "train" : DataLoader(train_set,shuffle = True, batch_size=batch_size),
      "test" : DataLoader(val_set,shuffle = False, batch_size = batch_size)
  }

  # Model
  device = torch.device('cuda')
  model = EfficientNetModel(num_classes=18).to(device)

  model_path = './checkpoint/'+model.model_name+'_'+get_local_time(date=False)+'_checkpoint.pt'

  # Hyperparameters
  LEARNING_RATE = 0.001

  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  # base_optimizer = torch.optim.SGD
  # optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, lr=LEARNING_RATE)

  loss_weight = get_loss_weight(train_csv, beta=0.999).to(device)
  loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

  # Train Model
  print('Model Training...')
  model, _, _ = train(device, model, dataloaders, optimizer, loss_fn, model_path, NUM_EPOCH=50, patience=5)

  # Predict Test Label
  submission(test_path, transform, model, device)
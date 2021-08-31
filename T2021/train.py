from importlib import import_module
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A
import albumentations.pytorch

from utils import *
from model import *

import argparse

import warnings
warnings.simplefilter("ignore", UserWarning)

def train(args):
  
  # settings
  device = torch.device('cuda')
  wandb.init(project='FaceMaskClassification', entity='dainkim')

  # dataset
  train_img = '../input/data/train/images'
  
  dataset_module = getattr(import_module("data.dataset"), args.dataset)
  dataset = dataset_module(
    data_dir=train_img,
  )

  # augmentation
  transform = transforms.Compose([
          transforms.CenterCrop(350),
          transforms.Resize((224, 224), Image.BILINEAR),
          transforms.ToTensor(),
        ])

  albumentations_transform = A.Compose([
    A.CenterCrop(350, 350),
    A.Resize(224, 224),
    A.RandomBrightnessContrast(),
    A.GaussNoise(p=0.2),
    A.pytorch.transforms.ToTensorV2()
  ])

  dataset.set_transform(transform)
  
  # data loader
  train_set, val_set = dataset.split_dataset()

  dataloaders = {
      "train" : DataLoader(train_set,shuffle = True, batch_size= args.batch_size),
      "test" : DataLoader(val_set,shuffle = False, batch_size = args.valid_batch_size)
  }

  # model
  model_module = getattr(import_module("model.model"), args.model)  # default: BaseModel
  if args.model=='EfficientNetMSD':
    model = model_module(
      num_classes = 18,
      sample_size = args.sample_size
    ).to(device)

  else:
    model = model_module(
      num_classes=18
    ).to(device)
  
  # loss & metric
  criterion = create_criterion(args.criterion)  # default: cross_entropy

  if args.beta > 0: #weighted cross entropy
    # class_list = [2234, 1619, 341, 2942, 3223,  423, 450, 321, 66, 589, 640, 86, 453, 323, 68, 592, 664, 86]
    class_list = get_label_count(train_set, num_classes=18)
    if torch.is_tensor(class_list): class_list = class_list.tolist()
    loss_weight = get_loss_weight(class_list, beta=args.beta).to(device)
    criterion = create_criterion(args.criterion, weight=loss_weight)

  opt_module = getattr(import_module("torch.optim"), args.optimizer)
  optimizer = opt_module(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=5e-4
  )
    
  # early stopping
  model_path = './checkpoint/'+model.model_name+'_'+get_local_time()+'_checkpoint.pt'
  early_stopping = EarlyStopping(patience = args.patience, path=model_path)

  # Start Training
  print('Model Training...')

  for epoch in range(args.epochs):
    for phase in ["train", "test"]:
      running_loss = 0.
      running_acc  = 0.
      running_f1 = 0.
      n_iter = 0

      if phase == "train":
        model.train()
      elif phase == "test":
        model.eval()

      for _, (images, labels) in enumerate(tqdm(dataloaders[phase])):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase=="train"):
          outputs = model(images)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == "train":
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
        running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
        running_f1 += f1_score(labels.data.cpu(), preds.cpu(), average='weighted')
        n_iter += 1

      # 한 epoch이 모두 종료되었을 때,
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      epoch_acc = running_acc / len(dataloaders[phase].dataset)
      epoch_f1 = running_f1/ n_iter

      if phase=="train":
        wandb.log({"train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "train_f1": epoch_f1})

      if phase=="test":
        wandb.log({"test_loss": epoch_loss,
                    "test_acc": epoch_acc,
                    "test_f1": epoch_f1})
        early_stopping(epoch_loss, model)

      print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, f1 score:{epoch_f1:.3f}")
      
    if early_stopping.early_stop:
      print("Early stopping")
      break
  
  print("학습 종료!")
  model.load_state_dict(torch.load(model_path))
  print(model_path)

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
  
  parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')  
  
  parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
  parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')

  parser.add_argument('--model', type=str, default='EfficientNetModel', help='model type (default: BaseModel)')
  parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
  parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
  parser.add_argument('--lr', type=int, default=0.001, help='learning rate (default: 0.001)')

  parser.add_argument('--patience', type=int, default=5, help='early stopping patience(default: 5)')

  parser.add_argument('--beta', type=int, default=0.999, help='calculating cross entropy loss weight (default: 0.999)')
  parser.add_argument('--sample_size', type=int, default=2, help='dropout sample size for MSD (default: 2)')

  args = parser.parse_args()
  
  # Train Model
  train(args)
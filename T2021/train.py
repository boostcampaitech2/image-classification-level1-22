import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils import *

def train(device, model, dataloaders, optimizer, loss_fn, model_path, SAM=False, NUM_EPOCH=10, patience=3):
  wandb.init(project='FaceMaskClassification', entity='dainkim')
  
  train_losses = []
  train_accs = []

  early_stopping = EarlyStopping(patience = patience, path=model_path)

  for epoch in range(NUM_EPOCH):
    for phase in ["train", "test"]:
      running_loss = 0.
      running_acc  = 0.
      y_true = []
      y_pred = []

      if phase == "train":
        model.train()
      elif phase == "test":
        model.eval()

      for _, (images, labels) in enumerate(tqdm(dataloaders[phase])):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase=="train"):
          outputs = model(images)
          _, preds = torch.max(outputs, 1)
          loss = loss_fn(outputs, labels)

          if phase == "train":
            loss.backward()

            if SAM:
              optimizer.first_step(zero_grad=True)
              loss_fn(model(images), labels).backward()
              optimizer.second_step(zero_grad=True)

            else:
              optimizer.step()
            
        running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
        running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장
        y_true.append(labels.data)
        y_pred.append(preds)


      # 한 epoch이 모두 종료되었을 때,
      y_true = y_true[0].tolist()
      y_pred = y_pred[0].tolist()
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      epoch_acc = running_acc / len(dataloaders[phase].dataset)
      epoch_f1 = f1_score(y_true, y_pred, average='weighted')

      if phase=="train":
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        wandb.log({"train_loss": epoch_loss})
        wandb.log({"train_acc": epoch_acc})
        wandb.log({"train_f1": epoch_f1})

      if phase=="test":
        wandb.log({"test_loss": epoch_loss})
        wandb.log({"test_acc": epoch_acc})
        wandb.log({"test_f1": epoch_f1})
        early_stopping(epoch_loss, model)

      print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, f1 score:{epoch_f1:.3f}")
      
    if early_stopping.early_stop:
      print("Early stopping")
      break
  
  print("학습 종료!")
  model.load_state_dict(torch.load(model_path))

  return model, train_losses, train_accs

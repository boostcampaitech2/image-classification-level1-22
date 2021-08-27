import torchvision
import torch
import os
import time
import wandb
import albumentations as A
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
from torchvision import transforms
from sklearn.metrics import f1_score
import torch.optim as optm
import torch.functional as F
import tqdm
from pytz import timezone
from datetime import datetime as dt

from data.dataset import Mask_Dataset
from model.model import ViTBase16, ResNet50, ViTBase32, R50ViT, Efficientnet

device = torch.device("cuda")
torch.manual_seed(42)

wandb.init(project="pstage-image", entity="ththth663")

TRAIN_CSV_PATH = "/opt/ml/code/splitted_train_58.csv"
VALID_CSV_PATH = "/opt/ml/code/splitted_valid_58.csv"

EPOCHS = 20
CLASS_NUM = 18
BATCH_SIZE = 16
LEARNING_RATE = 0.001
# FREEZE_TRAINED_LAYERS = 0


def get_dataLoader():
    transforms = torchvision.transforms.Compose(
        [
            CenterCrop((384, 384)),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = Mask_Dataset(
        TRAIN_CSV_PATH,
        transform=transforms,
        train=True,
    )

    valid_dataset = Mask_Dataset(
        VALID_CSV_PATH,
        transform=transforms,
        train=True,
    )

    train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataLoader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataLoader, valid_dataLoader


def train(
    model, optimizer, loss_fn, epochs, batch_size, train_dataLoader, valid_dataLoader
):

    model.to(device)
    
    # wandb 설정
    wandb.watch(model)
    wandb.run.name = "Efficient_Net_58"

    for epoch in range(epochs):
        running_loss, running_acc = 0.0, 0.0
        model.train()
        pbar = tqdm.tqdm(
            enumerate(train_dataLoader),
            total=len(train_dataLoader),
            position=True,
            leave=True,
        )

        for idx, (images, labels) in pbar:
            epoch_f1 = 0
            images = images.to(device)
            labels = labels.to(device).long()
            hypothesis = model(images)
            cost = loss_fn(hypothesis, labels)

            _, preds = torch.max(hypothesis, 1)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            running_loss += cost.item() * images.shape[0]
            running_acc += torch.sum(preds == labels.data)

        running_acc = running_acc / (len(train_dataLoader) * batch_size)

        model.eval()
        correct = 0
        pbar = tqdm.tqdm(
            enumerate(valid_dataLoader),
            total=len(valid_dataLoader),
            position=True,
            leave=True,
        )

        # 검증
        for idx, (valid_image, labels) in pbar:
            epoch_f1 = 0

            with torch.no_grad():
                images = valid_image.to(device)
                preds = model(images)
                preds = preds.argmax(dim=-1)
                correct += torch.sum(preds.cpu() == labels.data)
                epoch_f1 += f1_score(
                    preds.cpu().numpy(), labels.cpu().numpy(), average="weighted"
                )
                epoch_f1 = epoch_f1 / (idx + 1)

            valid_accuracy = correct / (len(valid_dataLoader) * BATCH_SIZE)

        wandb.log(
            {
                "loss": running_loss,
                "train_accuracy": running_acc,
                "valid_accuracy": valid_accuracy,
                "f1_score": epoch_f1,
            }
        )

        print(
            "[Epoch: {:>4}] cost = {:>.5} acc = {:>.5} val = {:>.5} f1_score = {:>.5}".format(
                epoch + 1, running_loss, running_acc, valid_accuracy, epoch_f1
            )
        )

        now = dt.now().astimezone(timezone("Asia/Seoul")).strftime('%Y%m%d%H%M%S')
        torch.save(
            model, os.path.join("/opt/ml/models", f"58_vit16_{now}_{valid_accuracy}.pt")
        )


if __name__ == "__main__":

    # model = torch.load("/opt/ml/models/efficient_Thu Aug 26 04:12:29 2021_4028.046142578125.pt")
    model = ViTBase16(CLASS_NUM, pretrained=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optm.Adam(model.parameters(), lr=LEARNING_RATE)
    train_dataLoader, valid_dataLoader = get_dataLoader()

    train(
        model,
        optimizer,
        loss_fn,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        train_dataLoader,
        valid_dataLoader,
    )

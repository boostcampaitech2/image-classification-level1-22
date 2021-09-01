import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset import MaskBaseDataset
from loss import create_criterion

import argparse
from loadconfig import json_to_config
from get_confusion_matrix import GetConfusionMatrix



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, config):
    
    seed_everything(config["seed"])

    if config["wandb"]:
        wandb.init(project="pstage-image", entity="ththth663")

    save_dir = increment_path(os.path.join(model_dir, config["name"]))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), config["dataset"]
    )  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), config["augmentation"]
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=config["resize"],
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["valid_batch_size"],
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), config["model"])  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)

    ######
    # 파라미터 잠그기
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.model.classifier_final.parameters():
    #     param.requires_grad = True
    ######

    model = torch.nn.DataParallel(model)

    if config["wandb"]:
        wandb.watch(model)
        name = config["name"]
        wandb.run.name = f"{name}"

    # -- loss & metric
    criterion = create_criterion(config["criterion"])  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), config["optimizer"])  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=5e-4,
    )
    scheduler = StepLR(optimizer, config["lr_decay_step"], gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(config["epochs"]):

        ####### confusion matrix
        label_cm = GetConfusionMatrix(  # <------------------<<<<
            save_path='confusion_matrix_image',
            current_epoch=epoch,  # 구분점을 epoch으로 두었습니다. (반드시 Epoch일 필요 X)
            n_classes=18,  # 클래스 개수 조정이 가능합니다.
            # labels=[_ for _ in range(1, 19)]  # (default=None) None인 경우 0 부터 17까지 생성됩니다. 만약 본인 label이 1~18 인 경우 [_ for _ in range(1, 19)] 로 반드시 수정해주세요.
            tag='example',
            # image_name='confusion_matrix',  # default file name
            # 파일명이 f"example.confusion_matrix.epoch{epoch}.png"로 저장됩니다.
            # only_wrong_label=False,  # wrong label만 표현합니다. (default: True)
            # count_label=True,  # 수량으로 표현합니다.(default: False)
            # savefig=False,  # 이미지를 저장합니다. (default: True)
            # showfig=True,  # for jupyter-notebook (default: False)
            figsize=(13, 12),  # <- default figsize
            # dpi=200,  # Matplotlib's default is 150 dpi. (default: 200)
            vmax=None)  # A max value of colorbar of heatmap
        ########

        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            if type(inputs) is dict:
                inputs = inputs["image"]
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % config["log_interval"] == 0:
                train_loss = loss_value / config["log_interval"]
                train_acc = matches / config["batch_size"] / config["log_interval"]
                current_lr = get_lr(optimizer)
                tmp = config["epochs"]
                print(
                    f"Epoch[{epoch}/{tmp}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                if config["wandb"]:
                    wandb.log({"train_loss": train_loss, "train_acc": train_acc})

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                if type(inputs) is dict:
                    inputs = inputs["image"]
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                label_cm.collect_batch_preds(labels, preds)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=config["dataset"] != "MaskSplitByProfileDataset",
                    )
                    
            label_cm.epoch_plot() 

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()

            if config["wandb"]:
                wandb.log({"valid_loss": val_loss, "valid_accuracy": val_acc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="./config.json", type=str, help='config.json path (default: "./config.json")')
    config = json_to_config(parser)

    args = parser.parse_args()
    print(args)

    data_dir = config["data_dir"]
    model_dir = config["model_dir"]

    train(data_dir, model_dir, config)

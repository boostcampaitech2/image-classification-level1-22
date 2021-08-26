import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from vision_transformer_pytorch import VisionTransformer

import timm
from pytorch_pretrained_vit import ViT

class Ensemble_model(nn.Module):
    '''eval 전용?'''
    def __init__(self):
        super(Ensemble_model, self).__init__()
        self.model_age = torch.load("/opt/ml/model_only_age/efficient_Thu Aug 26 05:08:09 2021_0.9889599084854126.pt")
        self.model_jender = torch.load("/opt/ml/model_only_jender/efficient_Thu Aug 26 05:44:23 2021_0.9956563711166382.pt")
        self.model_mask = torch.load("/opt/ml/model_only_mask/efficient_Thu Aug 26 06:17:19 2021_0.9958373308181763.pt")

    def forward(self, x):
        age = self.model_age(x)
        jender = self.model_jender(x)
        mask = self.model_mask(x)
        return age + (jender * 3) + (mask*6)


class R50ViT(nn.Module): # input size = (384, 384)
    def __init__(self, n_classes, pretrained=False):
        super(R50ViT, self).__init__()
        self.r50vit = VisionTransformer.from_pretrained('R50+ViT-B_16')
        self.r50vit.classifier = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self.r50vit(x)
        return x


class ViTBase16(nn.Module): # input size = (384, 384)
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.vit16 = ViT("B_16_imagenet1k", pretrained=pretrained)
        self.vit16.fc = nn.Linear(768, n_classes, bias=True)

    def forward(self, x):
        x = self.vit16(x)
        # x = self.layer1(x)
        return x


class ViTBase32(nn.Module):
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase32, self).__init__()

        # self.layer0 = nn.Sequential(*list(ViT("B_16_imagenet1k", pretrained=pretrained).children())[0:-1])
        self.vit32 = ViT("B_32_imagenet1k", pretrained=pretrained)
        self.vit32.fc = nn.Linear(768, n_classes, bias=True)

    def forward(self, x):
        x = self.vit32(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.resnet50 = models.wide_resnet50_2(pretrained=pretrained)
        self.resnet50.fc = nn.Linear(768, n_classes, bias=True)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(1000, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class Efficientnet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.model= timm.create_model('efficientnet_b0', num_classes= num_classes, pretrained=pretrained)

    def forward(self, x):
        x= self.model(x)
        return x


if __name__ == "__main__":
    model = R50ViT(18, True)
    print(model)

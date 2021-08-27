import torch
import torch.nn as nn

from torchvision import models
from vision_transformer_pytorch import VisionTransformer

import timm
from pytorch_pretrained_vit import ViT


class Ensemble_model(nn.Module):
    def __init__(self, model1_path, model2_path, model3_path):
        super(Ensemble_model, self).__init__()
        self.model_age = torch.load(model1_path)
        self.model_jender = torch.load(model2_path)
        self.model_mask = torch.load(model3_path)

    def forward(self, x):
        age = self.model_age(x)
        jender = self.model_jender(x)
        mask = self.model_mask(x)
        return age + (jender * 3) + (mask * 6)


class R50ViT(nn.Module):  # input size = (384, 384)
    """pretrained only"""

    def __init__(self, n_classes):
        super(R50ViT, self).__init__()
        self.r50vit = VisionTransformer.from_pretrained("R50+ViT-B_16")
        self.r50vit.classifier = nn.Linear(
            in_features=768, out_features=n_classes, bias=True
        )

    def forward(self, x):
        x = self.r50vit(x)
        return x


class ViTBase16(nn.Module):  # input size = (384, 384)
    def __init__(self, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.vit16 = ViT("B_16_imagenet1k", pretrained=pretrained)
        self.vit16.fc = nn.Linear(768, n_classes, bias=True)

    def forward(self, x):
        x = self.vit16(x)
        return x


class ViTBase32(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase32, self).__init__()
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
        self.model = timm.create_model(
            "efficientnet_b0", num_classes=num_classes, pretrained=pretrained
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = R50ViT(18, True)
    for i in model.named_children():
        print(i)

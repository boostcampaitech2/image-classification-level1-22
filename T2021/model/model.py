import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

from torchvision import models

from efficientnet_pytorch import EfficientNet
  
class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = 'resnet50'
        self.num_classes = num_classes
        self.model = models.resnet50(pretrained=True)
        self.num_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_fc, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = 'efficientnet-b0'
        self.num_classes = num_classes

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.l2 = nn.Linear(256, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        # x = self.activation(x)
        x = self.l2(x)
        
        return x


# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000):
#         super(MyModel, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(64, 32),
#             nn.ReLU(inplace=True),
#             nn.Linear(32, num_classes),
#         )
# 
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
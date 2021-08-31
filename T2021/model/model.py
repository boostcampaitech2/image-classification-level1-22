import torch
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

        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class EfficientNetMSD(nn.Module):
    def __init__(self, num_classes, sample_size):
        super().__init__()
        self.model_name = 'efficientnet-b0-multidropout'
        self.num_classes = num_classes
        self.sample_size = sample_size

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.sample_size)])
        self.fc = nn.Linear(1000, self.num_classes)

        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)

        for i, dropout in enumerate(self.dropouts):
            if i==0:
                out = dropout(x)
                out = self.fc(out)

            else:
                temp_out = dropout(x)
                out += self.fc(temp_out)
        return torch.sigmoid(out/len(self.dropouts))
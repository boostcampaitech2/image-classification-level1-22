import torch.nn as nn
import timm

class MyModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.name = 'efficientnet_b0'
        self.model = timm.create_model(self.name, num_classes=18, pretrained=pretrained)
        self.name += f'_{pretrained}'

    def forward(self, X):
        return self.model(X)
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= timm.create_model('efficientnet_b0', num_classes= 18, pretrained=True)

    def forward(self, data):
        data= self.model(data)

        return data
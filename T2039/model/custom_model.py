import math
import torch
import torch.nn as nn
import torchvision
import timm

class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.name = 'resnet18'
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.weight.size(1), out_features=num_classes, bias= True)
        # initialize weight and bias
        #torch.nn.init.xavier_uniform_(self.model.fc.weight)
        #stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        #self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()    
        self.name = 'efficientNet'
        self.model = timm.create_model('efficientnet_b0', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.model(x)

class CustomVit(nn.Module):
    def __init__(self, num_classes):
        super(CustomVit,self).__init__()
        self.name = 'ViT'
        self.model = timm.create_model('vit_base_patch16_224', num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.model(x)
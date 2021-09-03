import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import timm

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes= num_classes
        self.model= timm.create_model('efficientnet_b0', num_classes= self.num_classes, pretrained= True)
    
    def forward(self, data):
        data= self.model(data)

        return data

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super.__init__() # initialize

        self.conv1= torch.nn.Conv2D(in_channel= 3, out_channel=12, kernel_size= 7, stride= 1)
        self.conv2= torch.nn.Conv2d(in_channel= 12, out_channel= 24, kernel_size= 3, stride= 1)
        self.conv3= torch.nn.Conv2d(in_channel= 24, out_channel= 48, kernel_size= 3, stride= 1)
        self.fc= nn.Linear(48, num_classes)
    
    def forward(self, x):
        
        x= self.conv1(x)
        x= F.relu(x)
        x= self.conv2(x)
        x= F.relu(x)
        x= self.conv3(x)
        x= F.relu(x)
        x= x.view(-1, 128)


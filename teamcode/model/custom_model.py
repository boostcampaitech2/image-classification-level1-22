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

class CustomEfficientNetWithClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = 'efficientnet-b0'
        self.num_classes = num_classes

        self.model = timm.create_model('efficientnet_b0', num_classes=num_classes, pretrained=True)

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

# Add ResNet50
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.name = 'resnet50'
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.weight.size(1), out_features=num_classes, bias= True)

    def forward(self, x):
        return self.model(x)

### ------------DarkNet--------- ####
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class MyNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(MyNetBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class MyNet(nn.Module):
    def __init__(self, block, num_classes):
        super(MyNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

# config -> mynet
def mynet(num_classes):
    return MyNet(MyNetBlock, num_classes)
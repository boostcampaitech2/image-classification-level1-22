import torch
import torch.nn as nn
import torch.nn.functional as F
from vision_transformer_pytorch import VisionTransformer
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0", num_classes=num_classes, pretrained=pretrained
        )
        self.model.classifier = nn.Linear(1280, num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNet_self(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0", num_classes=num_classes, pretrained=pretrained
        )
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 4048, bias=True),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.model.classifier_final = nn.Linear(4048, 3, bias=True)
        self.load_state_dict(
            torch.load("/opt/ml/models/EfficientNet_self_supervised.pt")
        )
        self.model.classifier_final = nn.Linear(4048, num_classes, bias=True)

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


if __name__ == "__main__":
    model = EfficientNet_self(18, True)
    for i in model.model.named_children():
        print(i)

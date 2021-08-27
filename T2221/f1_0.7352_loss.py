
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.module import Module
import torch.nn as nn
import torch

class CustomLoss(nn.Module):
    def __init__(self, classes, epsilon= 1e-7):
        super().__init__()
        super(nn.CrossEntropyLoss).__init__()
        self.classes= classes
        self.epsilon= epsilon
    
    def forward(self, y_pred, y_true):

        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        CE= torch.nn.CrossEntropyLoss()

        CE_loss= CE(y_pred, y_true)
        Focal_loss= Focal(y_pred, y_true)

        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        # print(y_true.shape, y_pred.shape)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        return (1 - f1.mean()) + CE_loss 


if __name__== '__main__':

    loss= CustomLoss(18)
    loss.forward(torch.tensor([[0.01*i for i in range(1, 19)]for j in range(3)]), torch.tensor([1, 15, 17]))
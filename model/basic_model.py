import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torch

class BasicModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        return self.model(x)['out']

class BasicModel2(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    
    def forward(self, x):
        return self.model(x)['out']


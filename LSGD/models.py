import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

# Helper Class
class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

# 7-layer model from section 5.1 of EASGD paper (https://arxiv.org/pdf/1412.6651.pdf)
class EASGD_7_layer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._model = nn.Sequential(
            #32
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            #12
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            #4
            nn.Conv2d(128, 64, 3),
            nn.ReLU(True),
            #256
            View(),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)) #4

    def forward(self, x):
        return self._model(x)
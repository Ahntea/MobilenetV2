import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dws import DepthWiseSeparable as DWS
from mbconvblock import MBConvBlock as MBB

class MobilenetV2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.layer2 = nn.Sequential(
            MBB(32,16,multiple=1,stride=1),

            MBB(16,24,6,2),
            MBB(24,24,6,1),

            MBB(24,32,6,2),
            MBB(32,32,6,1),
            MBB(32,32,6,1),

            MBB(32,64,6,2),
            MBB(64,64,6,1),
            MBB(64,64,6,1),
            MBB(64,64,6,1),

            MBB(64,96,6,1),
            MBB(96,96,6,1),
            MBB(96,96,6,1),

            MBB(96,160,6,2),
            MBB(160,160,6,1),
            MBB(160,160,6,1),

            MBB(160,320,6,1),
        )
        self.layer3 = nn.Sequential(
        nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
        nn.BatchNorm2d(1280),
        nn.ReLU6(inplace=True)
        )
        self.avg = nn.AvgPool2d(7,7)
        self.linear = nn.Linear(1280, self.n_class)
        # self.linear = nn.Sequential(
        #     nn.Dropout(p=0.2)
        #     nn.Linear(1280, self.n_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg(x)
        x = x.view(-1, 1280)
        x = self.linear(x)
        return x

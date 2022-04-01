import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dws import DepthWiseSeparable as DWS
from mbconvblock import MBConvBlock as MBB

class EfficientNet(nn.Module):
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

            MBB(24,40,6,2),
            MBB(40,40,6,1,kernel_size=5, padding=2),
            MBB(40,40,6,1,kernel_size=5, padding=2),

            MBB(40,80,6,2),
            MBB(80,80,6,1),
            MBB(80,80,6,1),
            MBB(80,80,6,1),

            MBB(80,112,6,2),
            MBB(112,112,6,1,kernel_size=5, padding=2),
            MBB(112,112,6,1,kernel_size=5, padding=2),

            MBB(112,192,6,1,kernel_size=5, padding=2),
            MBB(192,192,6,1,kernel_size=5, padding=2),
            MBB(192,192,6,1,kernel_size=5, padding=2),

            MBB(192,320,6,1),
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

if __name__ == "__main__":
    effi = EfficientNet(10)
    tensor = torch.rand(1,3,244,244)
    print(effi(tensor))
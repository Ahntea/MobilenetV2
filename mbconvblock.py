import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dws import DepthWiseSeparable as DWS

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, multiple, stride):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.multiple = multiple
        self.stride = stride
        self.hidden = self.in_ch * self.multiple
        self.block = nn.Sequential(
            nn.Conv2d(self.in_ch , self.hidden, kernel_size=1,bias=False),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU6(inplace=True),
            DWS(self.hidden, self.out_ch, self.stride),
        )
    def forward(self, x):
        if self.stride == 1 and self.multiple != 1 and (self.in_ch == self.out_ch):
            res_x = x
            x = self.block(x)
            return x + res_x

        else:
            x = self.block(x)
            return x 

# mbcon1 = MBConvBlock(3,3,6,1)
# mbcon2 = MBConvBlock(3,3,6,2)
# tensor = torch.rand(1,3,244,244)
# print(mbcon1(tensor).shape)
# print(mbcon2(tensor).shape)

# depthwise = DWS(3,3,1)
# print(depthwise)
# tensor = torch.rand(1,3,244,244)
# print(tensor.shape)
# print(tensor)
# print("--------")
# print(depthwise(tensor))
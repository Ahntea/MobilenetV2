import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DepthWiseSeparable(nn.Module):
    
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        assert stride in [1,2]

        self.dw = nn.Sequential(
            nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1, stride=self.stride, groups=self.in_ch,bias=False),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
            # mobilenetV2 구현을 위해 비선형(ReLu) 제거
            # nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


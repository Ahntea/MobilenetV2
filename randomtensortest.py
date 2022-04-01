import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary as summary

from EfficientNet import EfficientNet
from MobilenetV2 import MobilenetV2
import torchvision.models as models

resnet18 = models.resnet18()
resnet18.to('cuda')
summary(resnet18,(3,224,224))

mobile2 = MobilenetV2(1000)
mobile2.to('cuda')
summary(mobile2,(3,224,224))

effi = EfficientNet(1000)
effi.to('cuda')
summary(effi,(3,224,224))

effib0 = models.efficientnet_b0()
effib0.to('cuda')
summary(effib0,(3,224,224))
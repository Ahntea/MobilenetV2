import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary as summary
from MobilenetV2 import MobilenetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
model = MobilenetV2(1000).to(device)
print(summary(model,(3,224,224)))


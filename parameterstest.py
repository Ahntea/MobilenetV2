import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from MobilenetV2 import MobilenetV2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MobilenetV2(10)
model.to(device)
model.load_state_dict(torch.load("/data/efficientnet/train/mobilenetv2.pt"))
model.eval()

for params in model.parameters():
    param = params.detach()
    cv2.imwrite("pram0", param[0])
    break
#     print("type : ", type(params), "size : ", params.size())


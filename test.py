import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import time

from MobilenetV2 import MobilenetV2

transformer = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transformer)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image

dataiter = iter(test_loader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MobilenetV2(10)
model.to(device)
model.load_state_dict(torch.load("/data/efficientnet/train/mobilenetv2.pt"))
model.eval()

print(model)

start = time.time()
predict = []
for images, labels in iter(test_loader):
    # print(images.shape)
    # temp=model(images.to(device))
    # predict.append(temp)
    model(images.to(device))
    break

print("mobilenet", time.time()-start)







start = time.time()
for images, labels in iter(test_loader):
    # print(images.shape)
    # temp=model(images.to(device))
    # predict.append(temp)
    model(images.to(device))
    break

print("resnet", time.time()-start)

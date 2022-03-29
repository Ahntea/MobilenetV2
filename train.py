import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from MobilenetV2 import MobilenetV2
from torchvision import datasets, transforms
from EarlyStopping import EarlyStopping

def im_convert(tensor):
  image = tensor.clone().detach().numpy()
  image = image.transpose(1, 2, 0)
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  image = image.clip(0, 1)
  return image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def get_model(device):
    model = MobilenetV2(10)
    model.to(device)
    return model

transformer = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=40, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

model = get_model(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 12
running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

early_stopping = EarlyStopping(patience = 7, verbose = True)

print("start training")
for e in range(epochs):

  running_loss = 0.0
  running_correct = 0.0
  validation_running_loss = 0.0
  validation_running_correct = 0.0
  batch=0
  for inputs, labels in training_loader:

    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)

    running_correct += torch.sum(preds == labels.data)
    running_loss += loss.item()

    if batch % 50 == 0:
    #   loss, current = loss.item(), batch * len(x)
    #   print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
      acc, current = running_correct/len(training_loader), batch * len(training_loader)
      print(f"batch_acc: {acc:>7f}  [{current/125:>5f}/{len(training_dataset):>5d}]")
    batch += 1
  else:
    # 훈련팔 필요가 없으므로 메모리 절약
    with torch.no_grad():

      for val_input, val_label in validation_loader:

        val_input = val_input.to(device)
        val_label = val_label.to(device)
        val_outputs = model(val_input)
        val_loss = criterion(val_outputs, val_label)

        _, val_preds = torch.max(val_outputs, 1)
        validation_running_loss += val_loss.item()
        validation_running_correct += torch.sum(val_preds == val_label.data)


    epoch_loss = running_loss / len(training_loader)
    epoch_acc = running_correct.float() / len(training_dataset)
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)

    val_epoch_loss = validation_running_loss / len(validation_loader)
    val_epoch_acc = validation_running_correct.float() / len(validation_dataset)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)

    print("===================================================")
    print("epoch: ", e + 1)
    print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
    print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))

    early_stopping(val_epoch_loss, model)

    if early_stopping.early_stop:
      print("Early stopping")
      break

torch.save((running_loss_history,validation_running_loss_history), "/data/efficientnet/train/loss_mobilenetv2.pt")
torch.save((running_correct_history, validation_running_correct_history),"/data/efficientnet/train/acc_mobilenetv2.pt")
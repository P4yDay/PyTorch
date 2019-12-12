from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch import nn, optim
import numpy as np

## 1ST TRANSFORMATION CONVERTS THE RAW DATA INTO TENSOR VARIABLES 
# AND THE SECOND TRANSFORMATION PERFORMS NORMALIZATION
_task = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5)), (())
])

## LOAD MNIST DATASET AND APPLY TRANSFORMATION
mnist = MNIST("data", download = True, train = True, transform =_task)

## CREATE TRAINING AND VALIDATION SPLIT
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

## create sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

## Create iterator objects for train and valid datasets
trainloader = DataLoader(mnist, batch_size = 256, sampler = tr_sampler)
validloader = DataLoader(mnist, batch_size = 256, sampler = val_sampler)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)
        return x
model = Model()

# Define the loss function and the optimizer using nn and optim package
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)

## Time to train the model * Insert Rocky Theme Song *
for epoch in range(1, 11):  # run model for 10 epochs
    train_loss, valid_loss = [], []
    ## Training part
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    ## evaluation part
    model.eval()
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
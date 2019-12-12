from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F 
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


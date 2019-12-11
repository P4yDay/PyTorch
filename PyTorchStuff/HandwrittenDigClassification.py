from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
## 1ST TRANSFORMATION CONVERTS THE RAW DATA INTO TENSOR VARIABLES 
# AND THE SECOND TRANSFORMATION PERFORMS NORMALIZATION
_task = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5)), (())
])

## LOAD MNIST DATASET AND APPLY TRANSFORMATION
 mnist = MNIST("data", download=True, train=True, transform=_task)
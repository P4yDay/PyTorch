from __future__ import print_function
import torch

# construct a 5x3 matrix
x = torch.empty(5,3)
print (x)

# construct a matrix filled with zerod and of dtype long
x = torch.zeros(5,3 , dtype=torch.long)
print(x)

# Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x)

# get its size
print(x.size())

# Operations

# adding the RIGHT way
y = torch.rand(5, 3)
print(x + y)

# or
print(torch.add(x, y))

# OR

result = torch.empty(5, 3)
torch.add(x,y, out=result) # prints the add to the result variable
print(result)

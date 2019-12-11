import torch

# creating  a simple three-layered network having 5 nodes in the input layer, 3 in the hidden layer, and 1 in the output layer.

n_input, n_hidden, n_output = 5, 3, 1

# init tensor for inputs and outputs

x = torch.randn((1, n_input))
y = torch.randn((1, n_output))

# init tensor variables for weights

w1 = torch.randn(n_input, n_hidden)
w2 = torch.randn(1, n_output)

# init tensor variables for bias terms

b1 = torch.randn((1, n_hidden)) # bias for hidden layers
b2 = torch.randn((1,n_output))  # bias for output layers

## CREATING A FORWARD PROPAGATION USING SIGMOID ACTIVATION
## FORMULA
## z = weight * input * bias
## a = activation_function(z)

def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))

# activation of hidden layer

z1 = torch.mm(x, w1) + b1
a1 = sigmoid_activation(z1)

# activation (output) of final layer

z2 = torch.mm(a1, w2) + b2
output = sigmoid_activation(z2)

## CALCULATING LOSS COMPUTATION
## Formula 
## loss = y - output
 
 ## MINIMIZE THE ERROR IN 
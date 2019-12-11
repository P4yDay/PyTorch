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
 
## MINIMIZE THE ERROR IN THE OUTPUT LAYER BY MAKING MARGINAL CHANGES IN BIAS AND WEIGHT
# fUNC TO CALCULATE THE DERIVATIVE OF ACTIVATION
def sigmond_delta(x):
    return x * (1 - x)

## compute derivate of error terms
delta_output = sigmond_delta(output)
delta_hidden = sigmond_delta(a1)

## backpass the changes to previous layers
d_outp = loss * delta_output
loss_h = torch.mm(d_outp, w2.t())
d_hidn = loss_h * delta_hidden

#UPDATING THE WEIGHTS AND BIAS USING DELTA CHANGES RECEIVED FROM THE ABOVE BACKPROP STEP
learning_rate = 0.2

w2 += torch.mm(a1.t(), d_outp) * learning_rate
w1 += torch.mm(x.t(), d_hidn) * learning_rate
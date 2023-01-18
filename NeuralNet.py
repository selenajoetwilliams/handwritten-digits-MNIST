''
OVERVIEW:

This is Selena's attempt at creating a neural network semi-from scratch
for the hand-written digits MNIST data set.

ABOUT THE DATASET: (QMNIST)

Note: I use the qmnit dataset which is the hand-written MNIST data set
with 60,000 images. It is part of the official torchvision datasets in 
pytorch documentation.

MAIN BUGS:

1. The labels for pyplot are messed up. Instead of indexing image 0 with
   label 0, every image has label 0 showing. I played around with this to
   try to figure it out so right now every image is labeled "happy" 
   instead of the correct #.

2. The accuracy is 100.0% every single time. Is this an error? 
   Note that the loss gets smaller each epoch. 

3. I haven't figured out how to properly push to github while using a 
   .gitignore file for the model_weights.pth file (which saves the model
   weights). I solved this by commenting out my code which saves & loads the
   model & deleting the model_weights.pth file. To save & load the model,
   simply uncomment the code & a model_weights.pth file will download. 
   question: How can I solve this? Should I use a .gitignore file whenever 
   uploading to github? I don't really want the file locally since it's just a 
   tutorial & since it takes up a lot of space.


NOTES:
I also have questions sprinkled throughout my code, noted with 'question'
'''

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models


###########################################
# TENSORS

training_data = datasets.QMNIST(
    root="data",
    train=True,
    download=True, 
    transform=ToTensor()
)

test_data = datasets.QMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# SAMPLING THE DATA VISUALLY

labels_map = {
    0: "happy",
    1: "one", 
    2: "two", 
    3: "three", 
    4: "four", 
    5: 5, 
    6: 6, 
    7: 7, 
    8: 8, 
    9: 9,
}

'''labels_map = {
    # NOTE: In the Fashion-Mnist example, I set these labels = to strings...
    # should it be the same for this?

    0: "test label",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "6", 
    7: "7",
    8: "8",
    9: "9",
}'''


# this sets ups a display of 9 random data in a 3x3 array
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

'''
BUG 1:

MY LABELS ARE ONLY SHOWING THE FIRST INDEX OF MY LABELS MAP
INSTEAD OF THE CORRECT LABELS (run the code to see)

Luckily I'm pretty sure that error is only with displaying the 
plot -- I don't think it will affect my network results.
'''
for i in range(1, cols * rows + 1): 
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i) # question: what does this line mean?
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()

###########################################
# TRANSFORMS 

ds = datasets.QMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor,
    target_transform=(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    # question: what does the line above mean?
)

###########################################
# BUILD MODEL
# this layer makes a neural network subclass in which the model is initialized and 
# the forward pass is defined 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # note: I am still using cpu 

class NeuralNetwork(nn.Module):

    # initializing the model
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # these are the different model layers
            nn.Linear(28*28, 512), # image dimensions, # of model dimentions?
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # defining the forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
# should the line above be: model = NeuralNetwork()
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

###########################################
# MODEL LAYERS

# this samples a mini batch to see what happens as it goes through the model
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# flattening the dimensions from 28x28 2D array to a 1D array of 784 pixels
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size)

# linear layer 
# this redcues the number of features from 784 to 20
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size)

# relu layer
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# sequential layer
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28) # question: why do we use torch.rand here?
logits = seq_modules(input_image)

# softmax layer
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# model parameters
# here we iterate over each parameter and print the size & a preview of its values
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
    # question: what does the param[:2] in the line above mean?

###########################################
# AUTOGRAD (automatic differentiation)

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# printing out the gradient function?
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# computing the gradients
print(f"\nComputing the gradients...")
loss.backward()
print(f"The gradient of the weights is: {w.grad}")
print(f"The gradient of the biasees is: {b.grad}")

###########################################
# OPTIMIZING MODEL PARAMETERS

# defining the hyper parameters
learning_rate = 1e-3
batch_size = 64
epochs = 2

# defining the type of loss function
# question: how is this different than the binary_cross_entropy_with_logits(z, y) 
# in the autograd section?
loss_fn = nn.CrossEntropyLoss

# defining the optimizer object
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# full implmenentation of optimization algorithm

# DEFINING THE TRAINING & TEST LOOP

# defining the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Computer prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            # question: what does this chunk of code mean?

# defining the test loop
'''
BUG 2: 

I'm not sure where or why this is happening, but MY ACCURACY IS 100.0%
EVERY TIME
'''
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")   

# initializing the loss function optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

###########################################
# SAVING AND LOADING MODEL

'''
BUG 3

I had to comment out this code because I don't want a model_weights.pth
file to download locally & because even if I did want it to, I don't know 
how to use a .gitignore file to push without including it.

question: How do I use .gitignore to push to git without uploading the 
model_weights.pth file?
'''

'''
# saving the model
model = models.vgg16(pretrained=True)
torch.save(model.state_dict, 'model_weights.pth')

# loading the model
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
'''

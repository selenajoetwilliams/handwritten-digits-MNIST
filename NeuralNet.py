'''
This is Selena's attempt at creating a neural network semi-from scratch
for the hand-written digits MNIST data set.

Note: I use the qmnit dataset which is the hand-written MNIST data set
with 60,000 images. It is part of the official torchvision datasets in 
pytorch documentation.
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
    # TODO: I am trying setting donwload=False to see if I can avoid git lfs issues
    transform=ToTensor()
)

test_data = datasets.QMNIST(
    root="data",
    train=False,
    download=True,
        # TODO: I am trying setting donwload=False to see if I can avoid git lfs issues
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
    # TODO: In the Fashion-Mnist example, I set these labels = to strings...
    # should it be the same for this?
    0: "Hey cutie",
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
TODO: ERORR: MY LABELS ARE ONLY SHOWING THE FIRST INDEX OF MY LABELS MAP
INSTEAD OF THE CORRECT LABELS (run the code to see)

Note: I am moving on for now. Luckily I'm pretty sure that error is only 
with showing the plot and it won't affect the neural network.
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
    # TODO: I have no idea what the line means
)

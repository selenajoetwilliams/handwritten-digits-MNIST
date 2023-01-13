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

labels_map = {
    # TODO: In the Fashion-Mnist example, I set these labels = to strings...
    # should it be the same for this?
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6", 
    7: "7",
    8: "8",
    9: "9",
}


# this sets ups a display of 9 random data in a 3x3 array
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1): 
    sample_idx = torch.randint(len(training_data), size=(1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i) # question: what does this line mean?
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()

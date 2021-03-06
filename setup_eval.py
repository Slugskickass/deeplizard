#import requests
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=1000
    ,shuffle=True
)
# The length of the data set
print(len(train_set))
# The targets of the data set
print(train_set.targets)

# This gets each peice of data
sample = next(iter(train_set))
print(len(sample))
image = sample[0]
label = sample[1]

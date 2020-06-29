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

display_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=10
)

batch = next(iter(display_loader))
print('len:', len(batch))
images, labels = batch
print('types:', type(images), type(labels))
print('shapes:', images.shape, labels.shape)

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
plt.show()
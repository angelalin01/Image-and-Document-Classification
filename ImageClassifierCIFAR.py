#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


# In[6]:


import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax

class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3072, 1000)
    # TODO
    # You need to add the second layer's parameters
        self.fc2 = torch.nn.Linear(1000, 10)
    def forward(self, X):
        batch_size = X.size(0)
    # This next line reshapes the tensor to be size (B x 3072)
    # so it can be passed through a linear layer.
        X = X.view(batch_size, -1)
    # TODO
    # You need to pass X through the two linear layers and relu
    # then return the final scores
    
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X


# In[5]:


class Convolutional(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3,
                                 out_channels=7,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)

    # You need to add the pooling, second convolution, and
    # three linear modules here
    self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)
    self.conv2 = torch.nn.Conv2d(in_channels=7,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)
    self.fc1 = torch.nn.Linear(2704, 130)
    self.fc2 = torch.nn.Linear(130, 72)
    self.fc3 = torch.nn.Linear(72, 10)

  def forward(self, X):
    batch_size = X.size(0)
    X = self.conv1(X)
    X = self.pool(X)
    X = self.conv2(X)
    X = F.relu(X)
    X = X.view(X.size(0), -1)
    X = self.fc1(X)
    X = F.relu(X)
    X = self.fc2(X)
    X = F.relu(X)
    X = self.fc3(X)
    X = torch.sigmoid(X)

    return X


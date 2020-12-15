import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import time

class BgNet(nn.Module):
    def __init__(self):
        super(BgNet, self).__init__()
        self.batch_size = 1

        self.out = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1)


    def forward(self, x):
        # Down
        x = self.out(x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
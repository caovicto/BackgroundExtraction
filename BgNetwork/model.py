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

        print("running network")
        in_channels = 34
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3),
            nn.UpsamplingBilinear2d(size=(360, 640))
        )

    def forward(self, x):
        x = self.model(x)
        x = self.upsample(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()



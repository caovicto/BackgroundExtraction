from tqdm import tqdm  # Displays a progress bar
import pkg_resources
import time

from dataloader import FrameDataset

pkg_resources.require("torchvision>=0.5.0")
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import BgNet

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for init_bkg, data, labels in tqdm(trainloader):
        # initialize input with initial background
        in_data = torch.cat((init_bkg, data), 1)

        in_data = in_data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        output = net(in_data)
        output = torch.where(output > 0, output, init_bkg)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()

    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))
    return running_loss


def validate(net, loader, device):  # Evaluate accuracy on validation / test set
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for init_bkg, data, labels in tqdm(loader):
            # initialize input with initial background
            in_data = torch.cat((init_bkg, data), 1)

            in_data = in_data.to(device)
            labels = labels.to(device)

            output = net(in_data)
            output = torch.where(output > 0, output, init_bkg)

            return output.detach().numpy()[0], labels.detach().numpy()[0]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    train_data = FrameDataset(flag='train', dataDir='../../dataset/test2/', data_range=(0, 1))
    train_loader = DataLoader(train_data, batch_size=1)

    lr = 0.001
    net = BgNet().to(device)
    # net.load('models/p3_0.001_1947')
    criterion = nn.MSELoss() #TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print('\nStart training')
    train_loss = []

    # for epoch in range(1): #TODO decide epochs
    #     print('-----------------Epoch = %d-----------------' % (epoch+1))
    #     tl = train(train_loader, net, criterion, optimizer, device, epoch+1)
    #     train_loss.append(tl)

    background, gt = validate(net, train_loader, device)
    background = np.rollaxis(background, 0, 3)
    gt = np.rollaxis(gt, 0, 3)

    cv2.imshow('', background / 255.0)
    cv2.waitKey(0)

    cv2.imshow('', gt / 255.0)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
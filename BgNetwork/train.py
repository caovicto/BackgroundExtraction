from tqdm import tqdm  # Displays a progress bar
import pkg_resources
import time

from dataloader import FrameDataset

pkg_resources.require("torchvision>=0.5.0")
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


from model import BgNet

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))
    return running_loss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    train_data = FrameDataset(flag='train', dataDir='../Files/SyntheticDataset/', data_range=(0, 1))
    # train_loader = DataLoader(train_data, batch_size=1)
    #
    # print(train_data)
    #
    # lr = 0.001
    # net = BgNet().to(device)
    # net.load('models/p3_0.001_1947')
    # criterion = nn.CrossEntropyLoss() #TODO decide loss
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #
    # print('\nStart training')
    # train_loss = []
    # val_loss = []
    #
    # for epoch in range(10): #TODO decide epochs
    #     print('-----------------Epoch = %d-----------------' % (epoch+1))
    #     tl = train(train_loader, net, criterion, optimizer, device, epoch+1)
    #     train_loss.append(tl)

    # print('\nFinished Training, Testing on test set')
    # test(test_loader, net, criterion, device)


if __name__ == "__main__":
    main()
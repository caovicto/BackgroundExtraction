import numpy as np
import os
import png
import torch
import re
import cv2
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image


def create_outgoing_mask(flow):
    """
    Generate a mask based on a input flow
    :param flow: The flow that is going to be used to create the mask
    :return: The mask
    """
    _, h, w = flow.shape
    flowu, flowv = flow[0:2]
    flowu = ((w - 1) >= flowu) & (flowu >= 0)
    flowv = ((w - 1) >= flowu) & (flowu >= 0)

    return np.logical_and(flowu, flowv)


class Data:
    def __init__(self, keyframe, frames, flows, masks):
        self.keyframe = keyframe
        self.masks = np.array(masks)

        self.diffs = []
        for frame in frames:
            self.diffs.append(np.abs(keyframe - frame))
        self.diffs = np.concatenate(self.diffs)

        self.warped_frames = []
        for i in range(len(frames)):
            warped = self.warp(frames[i], flows[i])
            warped = warped[0].numpy()
            self.warped_frames.append(warped)
        self.warped_frames = np.concatenate(self.warped_frames)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        x = x[np.newaxis, :]
        flo = flo[np.newaxis, :]
        x = torch.FloatTensor(x)
        flo = torch.FloatTensor(flo)

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def to_numpy(self):
        cat = []
        for i in range(0, len(self.masks)):
            cat.extend(self.warped_frames[i*3:(i+1)*3])
            cat.append(self.masks[i])
            cat.extend(self.diffs[i*3:(i+1)*3])
        cat = np.concatenate([self.keyframe, cat])

        return cat

    def initial_background(self):
        r, g, b = [], [], []
        for i in range(0, len(self.masks)):
            r.append(self.warped_frames[(i * 3)])
            g.append(self.warped_frames[(i * 3)+1])
            b.append(self.warped_frames[(i * 3)+2])

        r, g, b = np.mean(r, axis=0), np.mean(g, axis=0), np.mean(b, axis=0)
        return np.array([r, g, b])


class FrameDataset(Dataset):
    def __init__(self, flag, levels=2, dataDir='../dataset/', data_range=(0, 100)):
        assert (flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])
        print("load " + flag + " dataset start")
        print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))

        self.dataset = []

        folders = glob("{}/*/".format(dataDir))
        for i in range(data_range[0], data_range[1]):
            files = glob("{}/*".format(folders[i]))
            files = sorted(files)
            labels, frames = [], []
            flows = {0: [], 1: [], 2: [], 3: [], 4: []}
            masks = {0: [], 1: [], 2: [], 3: [], 4: []}

            for file in files:
                if re.search("f\d.png", file):  # frames
                    img = cv2.imread(file)
                    img = np.asarray(img).astype("f")
                    img = np.rollaxis(img, 2, 0)
                    frames.append(img)

                elif re.search("gt\d.png", file):  # labels
                    img = cv2.imread(file)
                    img = np.asarray(img).astype("f")

                    img = np.rollaxis(img, 2, 0)
                    labels.append(img)

                elif re.search("\.npy", file):  # flows
                    flow = np.load(file)
                    flow = np.rollaxis(flow, 2, 0)

                    key = file.split('/')[-1][0]
                    flows[int(key)].append(flow)
                    masks[int(key)].append(create_outgoing_mask(flow))

            # generate dataset
            for k in range(len(frames)):  # for each keyframe
                # Frames
                newFrames = frames[:]
                del newFrames[k]
                # Flows
                newFlows = flows[k]
                # Masks
                newMasks = masks[k]

                frameData = Data(frames[k], newFrames, newFlows, newMasks)
                self.dataset.append((frameData, labels[k]))

        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        init_background = data.initial_background()
        data = data.to_numpy()

        return torch.FloatTensor(init_background), torch.FloatTensor(data), torch.FloatTensor(label)

import numpy as np
import os
import png
import torch
import re

from glob import glob
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image


class FrameDataset(Dataset):
    def __init__(self, flag, dataDir='../dataset/', data_range=(0, 100)):
        assert(flag in ['train', 'eval', 'test', 'test_dev', 'kaggle'])
        print("load "+ flag+" dataset start")
        print("    from: %s" % dataDir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))

        self.dataset = []


        folders = glob("{}/*/".format(dataDir))
        for i in range(data_range[0], data_range[1]):
            files = glob("{}/*".format(folders[i]))
            labels = []
            flows = []
            frames = []

            for file in files:
                if re.search("f\d.png", file):
                    img = Image.open(file)
                    # Normalize input image
                    # .transpose(2, 0, 1) / 128.0 - 1.0
                    img = np.asarray(img).astype("f")
                    frames.append(img)

                elif re.search("gt\d.png", file):
                    img = Image.open(file)
                    labels.append(img)

                elif re.search("\.npy", file):
                    flow = np.load(file)
                    flows.append(flow)

            print(frames[0].shape, flows[0].shape)
            frameData = np.concatenate((frames, flows), axis=0)
            label = np.concatenate(labels)
            #
            # self.dataset.append((frameData, label))

        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]

        return torch.FloatTensor(img), torch.FloatTensor(label)

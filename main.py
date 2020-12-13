# from Utilities.flowlib import *
from Utilities.aligner import *
from Utilities import dataset

import torch

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # flows = []
    # for i in range(10):
    #     flow = read_flow('Files/SpyNet/out_{}.flo'.format(i))
    #     plt.imshow(flow)
    #     plt.savefig('Files/SpyNet/rawflow_{}.png'.format(i))
        # break
        # visualize_flow(flow, mode='RGB', save='Files/SpyNet/flow_{}.png'.format(i))
    # flow[flow == 0] = np.nan
    #     flows.append(flow)
    #
    # avgflow = np.median(flows, axis=0)
    # visualize_flow(avgflow, mode='RGB', save='Files/SpyNet/flow_average.png')



    for i in range(2):
        img = cv2.imread('Files/SyntheticDataset/tower.jpg')
        occ = cv2.imread('../dataset/occlusions/occlusion.png')
        newImg = dataset.transform(img)
        newImg = dataset.addOcclusion(newImg, occ)
        # cv2.imwrite('Files/SyntheticDataset/{}_occluded.png'.format(i), newImg)






if __name__ == "__main__":
    main()
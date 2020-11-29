import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def OpticalFlow(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(img1)
    mask[..., 1] = 255
    flow = cv.calcOpticalFlowFarneback\
        (gray1, gray2, flow=None,
        pyr_scale=0.5, levels=4, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)

    mag, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle*180/np.pi/2
    mask[..., 2] = cv.normalize(mag, None, 255, cv.NORM_MINMAX)

    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    cv.imshow("dense flow", rgb)
    plt.show()

img1 = cv.imread('../../pyflow/examples/car1.jpg')
img2 = cv.imread('../../pyflow/examples/car2.jpg')
OpticalFlow(img1, img2)

from Utilities.flowlib import *
from Utilities.aligner import *

import cv2
import numpy as np

def main():
    flows = []
    for i in range(10):
        flow = read_flow('Files/SpyNet/out_{}.flo'.format(i))
        # show_flow('Files/SpyNet/out_0.flo')
        # visualize_flow(flow, mode='RGB', save='Files/SpyNet/flow_{}.png'.format(i))
        # flow[flow == 0] = np.nan
        flows.append(flow)

    avgflow = np.median(flows, axis=0)
    visualize_flow(avgflow, mode='RGB', save='Files/SpyNet/flow_average.png')





if __name__ == "__main__":
    main()
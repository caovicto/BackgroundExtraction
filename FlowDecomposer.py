import tensorflow as tf
from tensorflow import keras

from FeatureExtractor import *
from LayerEstimator import *


class FlowDecomposer:
    def __init__(self, imgShape, levels=4):
        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers
        #
        ##################################
        self.featureExtractor = FeatureExtractor(imgShape)
        self.layerEstimator = LayerEstimator(imgShape)

        self.levels = levels

    def pyramid(self, input):
        ################################
        # TODO: feed forward input through all layers
        # TODO: output final info
        #
        ##################################
        for i in range(self.levels):
            pass
        pass

    def costVolume(self, frame1, frame2):
        ################################
        # TODO: Construct cost volume between two frames
        # TODO: search range of 4 pixels
        #
        ##################################
        pass

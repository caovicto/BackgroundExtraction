import tensorflow as tf
from tensorflow import keras


class LayerEstimator:
    def __init__(self, imgShape):

        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers, global pooling average, folly connected
        #
        ##################################
        self.model = keras.Sequential([
            keras.layers.conv2D()
        ])

    def feedforward(self, input):
        ################################
        # TODO: feed forward input through all layers
        # TODO: output final info
        #
        ##################################
        pass

    def backpropogation(self, loss):
        ################################
        # TODO: Update paraemeters based on
        #
        ##################################
        pass

    def tile(self, flowfields):
        ################################
        # TODO: Tile flow fields together for global motion vector
        #
        ##################################
        pass
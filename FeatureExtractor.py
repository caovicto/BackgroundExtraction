import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

class FeatureExtractor:
    def __init__(self, imgShape):
        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers
        #
        ##################################
        self.model = keras.Sequential([
            layers.Conv2D(128, (3, 3), (1, 1), 'same', activation='leaky relu'),
            layers.Conv2D(128, (3, 3), (1, 1), 'same', activation='leaky relu'),
            layers.Conv2D(96, (3, 3), (1, 1), 'same', activation='leaky relu'),
            layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='leaky relu'),
            layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='leaky relu'),
            layers.Conv2D(3, (3, 3), (1, 1), 'same', activation='leaky relu'),
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

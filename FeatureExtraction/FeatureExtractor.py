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

    def trainFusion(self, imgB, alpha, img,
                    img0, img1, img2, img3,
                    flow0, flow1, flow2, flow3,
                    level):
        """
        Fusion layer for feature extractor

        :param imgB:
        :param alpha:
        :param img:

        :param img0:
        :param img1:
        :param img2:
        :param img3:

        :param flow0:
        :param flow1:
        :param flow2:
        :param flow3:
        :param level:

        :return:
        :rtype:
        """
        ################################
        # TODO: feed forward input through all layers
        # TODO: output final info
        #
        ##################################
        with tf.variable_scope("FusionLayer_B_{}".format(level), reuse=tf.AUTO_REUSE):
            b, h, w = tf.unstack(tf.shape(img))



        pass



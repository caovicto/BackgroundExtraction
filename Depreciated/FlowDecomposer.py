import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU, Conv2D, AveragePooling2D, Dense

import itertools


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))

    grid_x = tf.reshape(tf.range(width), [1, 1, width])
    grid_x = tf.tile(grid_x, [num_batch, height, 1])
    grid_y = tf.reshape(tf.range(height), [1, height, 1])
    grid_y = tf.tile(grid_y, [num_batch, 1, width])

    flow_u, flow_v = tf.unstack(flow, 2, 3)
    pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
    pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
    inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                              pos_x >= 0.0)
    inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                              pos_y >= 0.0)
    inside = tf.logical_and(inside_x, inside_y)
    return tf.expand_dims(tf.cast(inside, tf.float32), 3)


class FlowDecomposer:
    def __init__(self, imgShape, learning_rate=0.001):
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers
        #
        ##################################
        self.FeatureExtractor = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
            Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.01),
        ])

        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers
        #
        ##################################
        self.FlowEstimator = Sequential([
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            LeakyReLU(alpha=0.3),
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            LeakyReLU(alpha=0.3),
            Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            LeakyReLU(alpha=0.3),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            LeakyReLU(alpha=0.3),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            LeakyReLU(alpha=0.3),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Dense(2)
        ])

        self.FeatureExtractor.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer = self.optimizer)
        self.FlowEstimator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                        optimizer=self.optimizer)

        self.H, self.W = imgShape
        self.searchRange = 4


    def costVolume(self, feat0, feat1):
        ################################
        # TODO: Construct cost volume between two frames
        # TODO: search range of 4 pixels
        #
        ##################################
        def getCost(feat0, feat1, shift):
            def pad(x, vpad, hpad):
                return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])
            def crop(x, vcrop, hcrop):
                return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

            v, h = shift
            vtop, vbottom, hleft, hright = max(v, 0), abs(min(v, 0)), max(h, 0), abs(min(h, 0))

            featPad0 = pad(feat0, [vbottom, vtop], [hleft, hright])
            featPad1 = pad(feat1, [vtop, vbottom], [hright, hleft])
            costPad = featPad0 * featPad1
            return tf.reduce_mean(crop(costPad, [vtop, vbottom], [hleft, hright]), axis=3)

        costLength = (2 * self.searchRange) ** 2
        cv = [0] * costLength
        depth = 0
        for v in range(-self.searchRange, self.searchRange+1):
            for h in range(-self.searchRange, self.searchRange+1):
                cv[depth] = getCost(feat0, feat1, [v, h])
                depth += 1

        cv = tf.stack(cv, axis=3)
        cv = tf.nn.leaky_relu(cv, 0.1)
        return cv



    def createFlow(self, imgs):
        """

        :param imgs: list of images, length of 4
        :type imgs: list
        :return: list of all flows
        :rtype: list
        """
        features = []

        for img in imgs:
            feature = self.FeatureExtractor(img)
            features.append(feature)

        flows = []
        for pair in itertools.combinations(features, 2):
            feat0, feat1 = pair
            cost = self.costVolume(feat0, feat1)
            x = tf.concat([feat1, cost], axis=3)
            x = self.FlowEstimator(x)

            flow = tf.expand_dims(tf.expand_dims(x, 1), 1)
            flow = tf.tile(flow, [1, self.H, self.W, 1])
            flows.append(flow)
        
        return flows

    def trainFeatureExtractor(self, train_x, train_labels, epochs=10, validation_split=0.2):
        self.FeatureExtractor.fit(
            train_x, train_labels,
            epochs=epochs, validation_split=validation_split,
            verbose=2
        )

    def trainFlowEstimator(self, train_x, train_labels, epochs=10, validation_split=0.2):
        self.FlowEstimator.fit(
            train_x, train_labels,
            epochs=epochs, validation_split=validation_split
        )

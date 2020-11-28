import tensorflow as tf
from tensorflow import keras
import tensorflow as tf

# from FeatureExtractor import *
# from LayerEstimator import *

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
    def __init__(self, imgShape, levels=4):
        ################################
        # TODO: Find dimensions of input data
        # TODO: 6 CNN layers
        #
        ##################################
        # self.featureExtractor = FeatureExtractor(imgShape)
        # self.layerEstimator = LayerEstimator(imgShape)

        self.levels = levels
        self.pyramidFilters = [16, 32, 64, 96]
        self.H, self.W = imgShape

        self.searchRange = 4

    def pyramid(self, x):
        ################################
        # TODO: feed forward x through all layers
        # TODO: output final info
        #
        ##################################
        with tf.variable_scope("FeaturePyramidExtractor", reuse=tf.AUTO_REUSE)
            for i in range(self.levels):
                x = tf.layers.Conv2D(self.pyramidFilters[i], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.pyramidFilters[i], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
            return x



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



    def TranslationEstimator(self, feature0, feature1):
        def convBlock(filters, kernel_size=(3, 3), strides=(1, 1)):
            x = tf.layers.Conv2D(filters, kernel_size, strides, 'valid')(x)
            x = tf.nn.leaky_relu(x, 0.2)
            return x

        with tf.variable_scope("TranslationEstimator", reuse=tf.AUTO_REUSE):
            cost = self.costVolume(feature0, feature1)
            x = tf.concat([feature0, cost], axis=3)
            
            x = convBlock(128, (3, 3), (1, 1))(x)
            x = convBlock(128, (3, 3), (1, 1))(x)
            x = convBlock(96, (3, 3), (1, 1))(x)
            x = convBlock(64, (3, 3), (1, 1))(x)
            
            feature = convBlock(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow0 = tf.layers.dense(x, 2)
            flow0 = tf.expand_dims(tf.expand_dims(flow0, 1), 1)
            flow0 = tf.tile(flow0, [1, self.H, self.W, 1])

            flow1 = tf.layers.dense(x, 2)
            flow1 = tf.expand_dims(tf.expand_dims(flow1, 1), 1)
            flow1 = tf.tile(flow1, [1, self.H, self.W, 1])
            
            return flow0, flow1

    def createFlow(self, img0, img1, img2, img3, img4):
        """
        
        :param img0: 
        :param img1: 
        :param img2: 
        :param img3: 
        :param img4: 
        :return: 
        :rtype: 
        """
        feature0 = self.pyramid(img0)
        feature1 = self.pyramid(img1)
        feature2 = self.pyramid(img2)
        feature3 = self.pyramid(img3)
        feature4 = self.pyramid(img4)
        
        flowF01, flowB01 = self.TranslationEstimator(feature0, feature1)
        flowF02, flowB02 = self.TranslationEstimator(feature0, feature2)
        flowF03, flowB03 = self.TranslationEstimator(feature0, feature3)
        flowF04, flowB04 = self.TranslationEstimator(feature0, feature4)
        
        flowF10, flowB10 = self.TranslationEstimator(feature1, feature0)
        flowF12, flowB12 = self.TranslationEstimator(feature1, feature2)
        flowF13, flowB13 = self.TranslationEstimator(feature1, feature3)
        flowF14, flowB14 = self.TranslationEstimator(feature1, feature4)

        flowF20, flowB20 = self.TranslationEstimator(feature2, feature0)
        flowF21, flowB21 = self.TranslationEstimator(feature2, feature1)
        flowF23, flowB23 = self.TranslationEstimator(feature2, feature3)
        flowF24, flowB24 = self.TranslationEstimator(feature2, feature4)

        flowF30, flowB30 = self.TranslationEstimator(feature3, feature0)
        flowF31, flowB31 = self.TranslationEstimator(feature3, feature1)
        flowF32, flowB32 = self.TranslationEstimator(feature3, feature2)
        flowF34, flowB34 = self.TranslationEstimator(feature3, feature4)

        flowF40, flowB40 = self.TranslationEstimator(feature4, feature0)
        flowF41, flowB41 = self.TranslationEstimator(feature4, feature1)
        flowF42, flowB42 = self.TranslationEstimator(feature4, feature2)
        flowF43, flowB43 = self.TranslationEstimator(feature4, feature3)
        
        return flowF01, flowF02, flowF03, flowF04, \
               flowF10, flowF12, flowF13, flowF14, \
               flowF20, flowF21, flowF23, flowF24, \
               flowF30, flowF31, flowF32, flowF34, \
               flowF40, flowF41, flowF42, flowF43, \
               flowB01, flowB02, flowB03, flowB04, \
               flowB10, flowB12, flowB13, flowB14, \
               flowB20, flowB21, flowB23, flowB24, \
               flowB30, flowB31, flowB32, flowB34, \
               flowB40, flowB41, flowB42, flowB43
        
        
        
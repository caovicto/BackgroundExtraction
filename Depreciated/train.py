from FlowDecomposer import *
import pathlib
import tensorflow as tf
from tensorflow import keras
import utils


def main():
    imgShape = (320, 192)

    train_df = utils.loadImages('../../De-fencing-master/dataset/Training Set/Training_Images/*0000*.jpg')
    train_labels = utils.loadImages('../../De-fencing-master/dataset/Training Set/Training_Labels/*0000*.png')
    train_df = train_df.squeeze()
    train_labels = train_labels.squeeze()

    flowDecomposer = FlowDecomposer(imgShape=imgShape)
    flowDecomposer.trainFeatureExtractor(train_df, train_labels)


if __name__ == "__main__":
    main()

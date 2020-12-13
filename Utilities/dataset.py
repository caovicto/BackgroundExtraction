"""
utils.py

Loading images and datasets
"""

import numpy as np
import random
import scipy as sp
from PIL import Image, ImageOps
from scipy import misc
import cv2
import glob
import tensorflow as tf
from progressbar import ProgressBar

# pbar = ProgressBar()


def loadDataset(folder):
    """

    :param folder:
    :type folder:
    :return:
    :rtype:
    """
    pass


def imread(filename):
    """
    Reads files
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    im = sp.misc.imread(filename)
    return im / 255.0


def imsave(np_image, filename):
    """Save image to file.
    Args:
    np_image: .
    filename: .
    """
    im = sp.misc.toimage(np_image, cmin=-1.0, cmax=1.0)
    im.save(filename)


def imwrite(filename, np_image):
    """Save image to file.
    Args:
    filename: .
    np_image: .
    """
    # im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
    im = sp.misc.toimage(np_image, cmin=-1.0, cmax=1.0)
    im.save(filename)


def imwrite_batch(filenames, np_images):
    """Save batch images to file.
    Args:
    filenames:
    """
    # TODO
    pass


def imresize(np_image, new_dims):
    """Image resize similar to Matlab.
    This function resize images to the new dimension, and properly handles
    alaising when downsampling.
    Args:
    np_image: numpy array of dimension [height, width, 3]
    new_dims: A python list containing the [height, width], number of rows, columns.
    Returns:
    im: numpy array resized to dimensions specified in new_dims.
    """
    # im = np.uint8(np_image*255)
    im = np.uint8((np_image + 1.0) * 127.5)
    im = Image.fromarray(im)
    new_height, new_width = new_dims
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    return np.array(im)


def loadImages(path, size=(320, 192)):
    X_data = []
    files = glob.glob(path)

    print("extracting {} files from ".format(len(files))+path)

    for myFile in files:
        image = cv2.imread(myFile)
        X_data.append(tf.cast(imresize(image, size), tf.float32))

    return np.array(X_data)


def transform(img):
    #################
    # TODO: homography transform img to create 4 other images
    ##############
    translate = np.array([
        [1, 0, random.uniform(-0.05, 0.05)],
        [0, 1, random.uniform(-0.05, 0.05)],
        [0, 0, 1]
    ])

    rotate = np.array([
        [np.cos(random.uniform(0, 0.05)), -np.sin(random.uniform(0, 0.05)), 0],
        [np.sin(random.uniform(0, 0.05)), np.cos(random.uniform(0, 0.05)), 0],
        [0, 0, 1]
    ])

    shear = np.array([
        [1, random.uniform(-0.05, 0.05), 0],
        [random.uniform(-0.05, 0.05), 1, 0],
        [0, 0, 1]
    ])

    Homography = translate.dot(rotate).dot(shear)

    warped_img = cv2.warpPerspective(img, Homography, img.shape[1::-1])
    # cv2.imshow('Warped Image', warped_img)
    # cv2.waitKey(0)
    return warped_img


def generate_dataset(dataFolder, occFolder):
    occList = glob.glob('{}/*.png'.format(occFolder))
    filelist = glob.glob('{}/*/*.jpg'.format(dataFolder))

    for infile in filelist:
        generate_frames(infile, random.choice(occList), dataFolder)


def generate_frames(imgPath, occPath, newPath, shape=(360, 640), num=5):
    file = imgPath.split('/')[-1]
    name = file.split('.')[0]

    for i in range(num):
        homographyPath = "{}/{}_f{}.png".format(newPath, name, i)
        save_homography(imgPath, shape, homographyPath)
        save_overlay(homographyPath, occPath, homographyPath)


def save_homography(file, shape, save):
    img = cv2.imread(file)
    newImg = transform(img)
    newImg = cv2.resize(newImg, (0,0), fx=2, fy=2)
    newImg = newImg[int(shape[0]*0.5):int(shape[0]*1.5), int(shape[1]*0.5):int(shape[1]*1.5)]

    cv2.imwrite(save, newImg)
    return newImg


def save_overlay(bg_file, fg_file, save):
    bg = Image.open(bg_file).convert('RGB')
    fg = Image.open(fg_file).convert('RGBA')
    fg = ImageOps.fit(fg, bg.size)

    bg.paste(fg, (0, 0), mask=fg)
    bg.save(save)

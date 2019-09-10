import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def prepareInputFromImageFolder(path):
    all_images = []
    for image_path in os.listdir(path):
        img = io.imread(path+image_path)
        img = resize(img, (148, 148))
        all_images.append(img)
    return np.array(all_images)


def prepareInputFromImage(path, count):
    all_images = []
    for image_path in os.listdir(path):
        img = io.imread(path+image_path)
        img = resize(img, (148, 148))
        if count != 0:
                all_images.append(img)
        count = count - 1
    return np.array(all_images)

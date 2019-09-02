import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np,os
import skimage.io as io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

def prepareInputFromImageFolder(path) :
  all_images = []
  for image_path in os.listdir(path):
    img = io.imread(path+image_path)
    img = resize(img,(148, 148))
    all_images.append(img)
  return np.array(all_images)
	

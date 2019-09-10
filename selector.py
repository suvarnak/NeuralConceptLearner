import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess_dataset import prepareInputFromImageFolder, prepareInputFromImage
import os


def load_target_image_one_shot(base_target_dir, concept):
    image_dir = base_target_dir + concept + "//"
    print("Loading target image from..." + image_dir)
    x_train = prepareInputFromImage(image_dir, 1)
    print(x_train.shape)
    # normalize
    print(np.max(x_train))  # should be 255
    x_train = x_train / np.max(x_train)
    print(np.max(x_train))


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def generate_target(source_model, base_target_dir, target_concept):
    path = base_target_dir + target_concept
    print(path)
    x_test = prepareInputFromImage(path, 1)
    print(x_test.shape)
    generated_img = source_model.predict(x_test)
    mse_on_target_generation = mse(x_test, generated_img)
    return mse_on_target_generation


def load_source_model(concept_name):
		modelpath = ".//source_models//"+concept_name+".h5"
		print(modelpath)
		autoencoder_model = tf.keras.models.load_model(modelpath)
		return autoencoder_model

def evaluate_source_models(eight_source_concepts, base_target_dir, target_concept):
	for concept in eight_source_concepts:
		source_model = load_source_model(concept)
		mse_on_target_generation = generate_target(source_model, base_target_dir, target_concept)
		print(mse_on_target_generation)
	return

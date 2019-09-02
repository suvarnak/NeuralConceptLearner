import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np,os
import skimage.io as io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras as keras 

def prepareInputFromImageFolder(path) :
  all_images = []
  for image_path in os.listdir(path):
    img = io.imread(path+image_path)
    img = resize(img,(148, 148))
    all_images.append(img)
  return np.array(all_images)

def autoencoder(input_img):
		#encoder
		#input = 150 x 150 x 3 (wide and thin)
		conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #150 x 150 x 32
		pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1) 
		conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #75 x 75 x 64
		pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2) 
		conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #36 x 36x 128 (small and thick)

		#decoder
		conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
		up1 = tf.keras.layers.UpSampling2D((2,2))(conv4) # 14 x 14 x 128
		conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
		up2 = tf.keras.layers.UpSampling2D((2,2))(conv5) # 28 x 28 x 64
		decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
		return decoded
def fit_model(autoencoder_model,train_X,valid_X,train_ground,valid_ground):
	batch_size = 128
	epochs = 30
	autoencoder_model_train_history = autoencoder_model.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
	loss = autoencoder_model_train_history.history['loss']
	val_loss = autoencoder_model_train_history.history['val_loss']
	epochs = range(epochs) 
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss -Autoencoder for Dog Images')
	plt.legend()
	plt.show()

def save_model(autoencoder_model,concept_name):
	autoencoder_model.save_model(concept_name+'.h5')


def build_model():
	# define architectur of conv. neural network
	inChannel = 3
	x, y = 148, 148
	input_img = tf.keras.layers.Input(shape = (x, y, inChannel))
	autoencoder_model = tf.keras.models.Model(input_img, autoencoder(input_img))
	autoencoder_model.compile(loss='mean_squared_error', optimizer = 'RMSprop')
	return autoencoder_model

def train_model(base_dir,concept):
	image_dir = base_dir + concept +"//"
	print("Loading images from..." +image_dir)  
	x_train = prepareInputFromImageFolder(image_dir)
	print(x_train.shape)   
	# normalize
	print(np.max(x_train)) # should be 255
	x_train = x_train / np.max(x_train)
	print(np.max(x_train))
	# train test split
	train_X,valid_X,train_ground,valid_ground = train_test_split(x_train,
																															x_train, 
																															test_size=0.2, 
																															random_state=13)
	autoencoder_model = build_model()
	fit_model(autoencoder_model,train_X,valid_X,train_ground,valid_ground)
	save_model(autoencoder_model,concept)
	return


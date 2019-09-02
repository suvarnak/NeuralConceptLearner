
import os
from learner import train_model

#Train the 8 source models
turn = 10
base_dir = "/content/few-shot/data/miniImageNet/images_background/"
# randomly pick 8 source concept labels for training
all_source_concepts = []
for folder in os.listdir(base_dir):
	all_source_concepts.append(folder)

eight_source_concepts = all_source_concepts[(turn-1)*8:(turn-1)*8+8]
for concept in eight_source_concepts:
	train_model(base_dir,concept)
#load target concept

#select most appropriate source model for target concept

#predict for target concept using discriminator source model trained on all 8 concepts 

 
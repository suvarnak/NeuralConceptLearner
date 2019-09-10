
import os
from learner import train_model
from selector import evaluate_source_models
#Train the 8 source models
turn = 1
#base_dir = "D:\\phd\\experiments\\few-shot\\data\\miniImageNet\\images_background\\"
base_dir = "/content/few-shot/data/miniImageNet/images_background/"
# sequentially pick 8 source concept labels for training
all_source_concepts = []
for folder in os.listdir(base_dir):
	all_source_concepts.append(folder)
eight_source_concepts = all_source_concepts[(turn-1)*8:(turn-1)*8+8]
print("Source models trained",eight_source_concepts)
for concept in eight_source_concepts:
  	train_model(base_dir,concept)
#load target concept
base_target_dir = "/content/few-shot/data/miniImageNet/images_evaluation/"
#eight_source_concepts=['n02108089','n02138441','n03017168','n03075370','n03584254','n03770439','n03838899','n03998194']
#base_target_dir = "D:\\phd\\experiments\\few-shot\\data\\miniImageNet\\images_evaluation\\"
all_target_concepts = []
for folder in os.listdir(base_target_dir):
	all_target_concepts.append(folder)

two_target_concepts = all_target_concepts[(turn-1)*2:(turn-1)*2+2]
print("Source concepts", eight_source_concepts)
for concept in two_target_concepts:
	print("Target:",concept)
	evaluate_source_models(eight_source_concepts,base_target_dir,concept)

#select most appropriate source model for target concept



#predict for target concept using discriminator source model trained on all 8 concepts 

 
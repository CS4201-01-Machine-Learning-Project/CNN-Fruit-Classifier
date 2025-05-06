import os
import re
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import operator
import time

classLabels = []
DEBUGGING = False
img_height = 100
img_width = 100
def extract_possible_fruit_names(folder_path):
    fruit_names = set()

    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if not os.path.isdir(full_path):
            continue

        cleaned_name = re.sub(r'\s*\d+$', '', entry).strip().lower()
        fruit_names.add(cleaned_name)

    return sorted(fruit_names)


def load_digit_images_from_folder(folder_path, image_size=(100, 100)):
    X = []
    y = []

    possible_fruit_names = extract_possible_fruit_names(folder_path)

    for subfolder in os.listdir(folder_path):
        specific_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(specific_path):
            continue

        label_name = re.sub(r'\s*\d+$', '', subfolder).strip().lower()

        if label_name not in possible_fruit_names:
            print(f"Skipping unknown fruit label: '{label_name}'")
            continue

        label = possible_fruit_names.index(label_name)

        for filename in os.listdir(specific_path):
            file_path = os.path.join(specific_path, filename)
            try:
                img = Image.open(file_path).convert('RGB').resize(image_size)
                X.append(np.array(img))
                y.append(label)
            except Exception as e:
                print(f"Could not load image {file_path}: {e}")

    return np.array(X), np.array(y)
testImages,testLabels = load_digit_images_from_folder(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Test"))
classLabels = extract_possible_fruit_names(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Test"))
testImages = (testImages/255.0).reshape(len(testImages),img_height,img_width,3)
for folder in os.listdir("models"):
	for file in os.listdir(os.path.join("models",folder)):
		if operator.contains(file,"keras"):
			print(file)
			model = keras.models.load_model(os.path.join("models",folder,file))
			loss,acc = model.evaluate(testImages,testLabels)
			predictions = model.predict(testImages).argmax(axis=1)
			truePredictions = dict()
			for target in classLabels:
				truePredictions[classLabels.index(target)]=0
			predictionsInClass = dict()
			classInDataset = dict()
			for i in range(len(predictions)):
				if testLabels[i] not in classInDataset:
					classInDataset[testLabels[i]]=1
				else:
					classInDataset[testLabels[i]]+=1
				if testLabels[i] not in predictionsInClass:
					predictionsInClass[testLabels[i]]=1
				else:
					predictionsInClass[testLabels[i]]+=1
				if predictions[i]==testLabels[i]:
						truePredictions[predictions[i]]+=1
			print("Recall for ", file)
			sum=0
			for target in classLabels:
				recall = (truePredictions[classLabels.index(target)]/classInDataset[classLabels.index(target)])
				print(target,":",recall)
				sum+=recall
			print(sum/len(classLabels))
			print("Precision for ",file)
			sum=0
			for target in classLabels:
				precision=(truePredictions[classLabels.index(target)]/predictionsInClass[classLabels.index(target)])
				print(target,":",precision)
				sum+=precision
			print(sum/len(classLabels))

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
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
fileToPredict="C:\\Users\\Aidan\\Documents\\Polymorph Games\\Foundation\\mods\\144827\\icons\\Orange.png"
img_height = 100
img_width = 100
image_size=(img_height,img_width)
def extract_possible_fruit_names(folder_path):
    fruit_names = set()

    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if not os.path.isdir(full_path):
            continue

        cleaned_name = re.sub(r'\s*\d+$', '', entry).strip().lower()
        fruit_names.add(cleaned_name)

    return sorted(fruit_names)
nonIndexedClassLabels=extract_possible_fruit_names(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Test"))
for folder in os.listdir("models"):
	for file in os.listdir(os.path.join("models",folder)):
		if operator.contains(file,"keras"):
			
			model = keras.models.load_model(os.path.join("models",folder,file))
			img = np.array([np.array(Image.open(fileToPredict).convert('RGB').resize(image_size))])
			
			img = (img/255.0).reshape(len(img),img_height,img_width,3)
			print(file,":",nonIndexedClassLabels[model.predict(img).argmax(axis=1)[0]])
				

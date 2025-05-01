import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import operator
import time
classLabels = []
DEBUGGING = True
img_height = 256
img_width = 256
def getImagesAndLabelsFromFolderPath(folderPath, image_size=(img_height, img_width)):
	X = []
	y = []
	for folder in os.listdir(folderPath):
		try:
			label = folder[0:operator.indexOf(folder,' ')].lower()
		except ValueError:
			try:
				label = folder[0:operator.indexOf(folder,'_')].lower()
			except:
				raise Exception('Bad filename',folder)
		if (label not in classLabels):
			classLabels.append(label)
		for file in os.listdir(os.path.join(folderPath,folder)):
			img = Image.open(os.path.join(folderPath, folder, file)).convert('RGB').resize(image_size)

			X.append(np.array(img))
			y.append(classLabels.index(label))
			if (DEBUGGING and len(X)%5==0):
				break
	return np.array(X), np.array(y)
testImages,testLabels = getImagesAndLabelsFromFolderPath(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Test"))
testImages = (testImages/255.0).reshape(len(testImages),img_height,img_width,3)
for folder in os.listdir("models"):
	imported = tf.saved_model.load(os.path.join("models",folder)
	
	print(str(folder)+":"+testAcc)

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import operator
def getImagesAndLabelsFromFolderPath(folder_path, image_size=(256, 256)):
    X = []
    y = []
    for folder in os.listdir(folder_path):
		try:
			label = int(folder[0:operator.indexOf(folder,' ')])
		except ValueError e:
			try:
				label = int(folder[0:operator.indexOf(folder,'_')])
		for file in os.listdir(folder):
			img = Image.open(os.path.join(folder_path, folder)).convert('L').resize(image_size)

			X.append(np.array(img))
			y.append(label)
    return np.array(X), np.array(y)
trainImages, trainLabels = getImagesAndLabelsFromFolderPath(os.path.join("fruits-36-original-size-main", "fruits-36-original-size-main", "Training"))
testImages, testLables =

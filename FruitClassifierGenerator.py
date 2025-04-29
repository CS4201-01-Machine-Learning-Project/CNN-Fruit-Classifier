import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import operator
DEBUGGING = True
classLabels = []
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
			if (DEBUGGING and len(X)>20):
				break
		
		
	return np.array(X), np.array(y)
trainImages, trainLabels = getImagesAndLabelsFromFolderPath(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Training"))
testImages, testLabels = getImagesAndLabelsFromFolderPath(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Test"))
validationImages, validationLabels = getImagesAndLabelsFromFolderPath(os.path.join("fruits-360-original-size-main", "fruits-360-original-size-main", "Validation"))
trainImages=trainImages/255.0
testImages=testImages/255.0
validationImages=validationImages/255.0
trainImages = trainImages.reshape(len(trainImages),img_height,img_width,3)
testImages = testImages.reshape(len(testImages),img_height,img_width,3)
validationImages = validationImages.reshape(len(validationImages),img_height,img_width,3)

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)])
model = models.Sequential([
	#keras.Input(shape=(None, 256, 256, 3)),
	data_augmentation,
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(64, activation='relu'),
	layers.Dense(len(classLabels), name="outputs")
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=10
history = model.fit(trainImages, trainLabels, batch_size=32, epochs=epochs, validation_data=(validationImages, validationLabels))
testLoss, testAcc = model.evaluate(testImages, testLabels)
print('Accuracy:',testAcc)
#print('Loss:',loss)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

with open('bestAccuracy.txt') as f:
	previousBestAcc = float(f.read())
if testAcc>= previousBestAcc:
	with open('bestAccuracy.txt','w') as f:
		f.write(str(testAcc))
	model.save('FruitClassifier.keras')
plt.show()

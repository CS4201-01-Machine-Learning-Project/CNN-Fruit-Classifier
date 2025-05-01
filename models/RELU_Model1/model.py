import re
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.python.client import device_lib

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

    return np.array(X), np.array(y), possible_fruit_names  

train_path = "../datasetCorrect/Training"
test_path = "../datasetCorrect/Test"
validation_path = "../datasetCorrect/Validation"

X_train, Y_train, _ = load_digit_images_from_folder(train_path)
X_test, Y_test, _ = load_digit_images_from_folder(test_path)
X_val, Y_val, _ = load_digit_images_from_folder(validation_path)

X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255

X_train = X_train.reshape((len(X_train), 100, 100, 3))
X_test = X_test.reshape((len(X_test), 100, 100, 3))
X_val = X_val.reshape((len(X_val), 100, 100, 3))

model = models.Sequential([

    layers.Conv2D(32,3,activation='relu',input_shape=(100,100,3)),

    layers.MaxPooling2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(256,activation='relu'),

    layers.Dense(78,activation='softmax')
])

model.compile(optimizer=optimizers.SGD(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_val, Y_val))

loss, acc = model.evaluate(X_test, Y_test)

print(acc)

model.save("../../dummy/RELU_Model1")
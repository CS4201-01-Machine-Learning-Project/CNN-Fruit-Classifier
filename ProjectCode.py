#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-------------------------------------------------------------------------
#srun -A bchn-delta-gpu --time=00:20:00 --nodes=1 --tasks-per-node=16 --partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=16g --pty /bin/bash
# Importing Python libraries
import re
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

def extract_possible_fruit_names(folder_path):
    fruit_names = set()

    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if not os.path.isdir(full_path):
            continue

        # Remove trailing number from folder name
        cleaned_name = re.sub(r'\s*\d+$', '', entry).strip().lower()
        fruit_names.add(cleaned_name)

    return sorted(fruit_names)


# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(100, 100)):
    X = []
    y = []

    # Automatically extract labels from subfolders
    possible_fruit_names = extract_possible_fruit_names(folder_path)

    for subfolder in os.listdir(folder_path):
        specific_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(specific_path):
            continue

        # Extract label name by removing trailing number
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

    return np.array(X), np.array(y), possible_fruit_names  # include labels list optionally

# Set your own paths here (relative to your project folder)
train_path = "./dataset/Training"
test_path = "./dataset/Test"

# Loading the raw images using the provided function. Hint: Use the provided load_digit_images_from_folder function that outputs X_train, Y_train for train_path and
# as X_test, Y_test for test_path
# --> add your Python code here
X_train, Y_train, label_names = load_digit_images_from_folder(train_path)
X_test, Y_test, _ = load_digit_images_from_folder(test_path)


# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]. Hint: divide them by 255
# --> add your Python code here
X_train = X_train / 255
X_test = X_test / 255


# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
# --> add your Python code here
X_train = X_train.reshape((len(X_train), 100, 100, 3))
X_test = X_test.reshape((len(X_test), 100, 100, 3))


# Building a CNN model
model = models.Sequential([

    # Add a convolutional layer with 32 filters of size 3x3, relu activation, and input shape 100x100x1
    # Use layers.[add a layer here],
    # --> add your Python code here
    layers.Conv2D(32,3,activation='relu',input_shape=(100,100,3)),
    # Add a max pooling layer with pool size 2x2
    # Use layers.[add a layer here],
    # --> add your Python code here
    layers.MaxPooling2D(pool_size=2),
    # Add a flatten layer to convert the feature maps into a 1D vector
    # Use layers.[add a layer here],
    # --> add your Python code here
    layers.Flatten(),
    # Add a dense (fully connected) layer with 64 neurons and relu activation
    # Use layers.[add a layer here],
    # --> add your Python code here
    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(256,activation='relu'),
    # Add the output layer with 10 neurons (digits 0–9) and softmax activation
    # Use layers.[add a layer here]
    # --> add your Python code here
    layers.Dense(10,activation='softmax')
])

# Compiling the model using optimizer = sgd, loss = sparse_categorical_crossentropy, and metric = accuracy
# --> add your Python code here
model.compile(optimizer=optimizers.SGD(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# Fitting the model with batch_size=32 and epochs=10
# --> add your Python code here
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

# Evaluating the model on the test set
loss, acc = model.evaluate(X_test, Y_test)

# Printing the test accuracy
# --> add your Python code here
print(acc)

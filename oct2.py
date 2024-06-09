import matplotlib.pyplot as plt 
import os 
import numpy as np
import cv2
import random
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop

NAME = "Macular-Degeneration"

# Import files
DATADIR = "/Users/dheeraj/Desktop/OCT/OCT2017/train"
CATEGORIES = ["CNV", "DME", "DRUSEN", "NORMAL"]
training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                final_path = os.path.join(path, img)
                img_array = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  
y = np.array(y)

X = X / 255.0

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

datagen.fit(X)


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CATEGORIES)))  # Assuming multi-class classification
model.add(Activation('softmax'))  # Using softmax for multi-class


model.compile(
    loss='sparse_categorical_crossentropy',  # For multi-class classification
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)


model.fit(datagen.flow(X, y, batch_size=32), epochs=50)

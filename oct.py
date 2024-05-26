#training data on kaggle.com
#import everything
import matplotlib.pyplot as plt 
import os 
import numpy as np
import cv2
import random
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

#name of h5 files 
NAME = "Macular-Degeneration"

#import files
DATADIR = "/Users/dheeraj/Desktop/OCT/OCT2017 /train"
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

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Specify the number of channels (1 for grayscale)
y = np.array(y)

X = X / 255.0

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # Converts 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=100, epochs=1, validation_split=0.3, callbacks=[tensorboard])

model.save('Mac_Degen-CNN.h5')
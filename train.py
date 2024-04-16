import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

from sklearn.model_selection import KFold
from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

DATADIR = "E:\\new_sodoku\\images\\"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
IMG_SIZE = 128

x = []
y = []
training_data = []

for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x = x.astype('float32')
x = x / 255.0

y = np.array(y)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs = 5)

model.save('model.h5')

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()

print("***********TRAINING***********")
print("###Images processed: " + str(len(training_data)) + " ###")
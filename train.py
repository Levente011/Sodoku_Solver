import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import configparser
from tqdm import tqdm
from sklearn.model_selection import KFold
from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

config = configparser.ConfigParser()
config.read('config.txt')

DATADIR = config.get('Paths', 'DATADROP')
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


training_data = []
IMG_SIZE = 100


# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_num = CATEGORIES.index(category)
#         for img in tqdm(os.listdir(path), desc=f"Processing {category}"):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 training_data.append([img_array, class_num])
#             except Exception as e:
#                 pass

def data_augmentation():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)

        for img_filename in (os.listdir(path)):
            try:
                
                img_array = cv2.imread(os.path.join(path,img_filename),cv2.IMREAD_GRAYSCALE)
                
                img_canny = cv2.Canny(img_array, 50, 50)
                
                contours, hierachy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    
                    if area > 5:
                        
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                        
                        x,y,w,h = cv2.boundingRect(approx)
                        
                        image_rect = img_array[y:y+h, x:x+w]
                        image_rect = cv2.resize(image_rect, (100, 100))
                        
                        kernel = np.ones((5,5), np.uint8)
                        
                        for blur_value in range(-30, 30):
                            
                            img = cv2.GaussianBlur(image_rect, (7,7), blur_value)
                            training_data.append([img, class_num])
                            
                            
                            
                            img_erosion = cv2.erode(img, kernel, iterations = 1)
                            img_erosion2 = cv2.erode(img, kernel, iterations = 2)
                            training_data.append([img_erosion, class_num])
                            training_data.append([img_erosion2, class_num])
                            
                            
                            
                            
                            img_dilation = cv2.dilate(img, kernel, iterations = 1)
                            img_dilation2 = cv2.dilate(img, kernel, iterations = 2)
                            training_data.append([img_dilation, class_num])
                            training_data.append([img_dilation2, class_num])
                            
            except Exception as e:
                raise(e)
            

# create_training_data()
data_augmentation()

print(len(training_data))

import random
random.seed(2700)
random.shuffle(training_data)


for features,label in training_data[:1]:
    print(features) 


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X.astype('float32')
X = X / 255.0

y = np.array(y)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs = 3)

model.save('model.h5')

print("***********TRAINING***********")
print("###Images processed: " + str(len(training_data)) + " ###")

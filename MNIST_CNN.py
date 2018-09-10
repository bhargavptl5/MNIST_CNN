# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:15:49 2018

@author: Bhargav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:23:08 2018

@author: Bhargav
"""
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

#normalizing the data
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
#one-hot encoding
n_classes = 10

print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
input_shape = (28,28,1)
#adding layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')

model.fit(X_train,Y_train,epochs=5,batch_size=32)
loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

i=0;
while(i<10):
    img = X_test[random.randint(1,10000)]
    
    img_class = model.predict_classes(img)
    prediction = img_class[0]
    
    classname = img_class[0]
    
    print("Class: ",classname)
    
    
    img = img.reshape((28,28))
    plt.imshow(img)
    plt.title(classname)
    plt.show()
    i=i+1
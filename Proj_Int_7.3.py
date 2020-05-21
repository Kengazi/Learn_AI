#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:18:11 2020

@author: kendalljohnson
"""

"Keras ANN"

"""
***** first :sudo pip3 install tensorflow **************
***** first :sudo pip3 install keras **************

Week 7 - A base in using data science with python

7.3 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

We are done with sklearn models and now will focus on Keras and Pytorch ANNs 

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

"""

# Imports
import numpy as np
#import matplotlib.pyplot as plt

# AI
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

""" 
This data is an 28x28 images of handwritten digits with changes in pixel intensities
"""

# Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing
X_test = X_test/ 255
X_train = X_train/ 255

# Changing to categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshaping to fit in keras neural net
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Main

# Way layers are added together 
model = Sequential()
# adding  model layers
model.add(Dense(50, activation='relu', input_shape=(28,28,1))) # regular hidden layer with 50 nodes but also take is the input shape
model.add(Dense(50, activation='relu'))# regular hidden layer with 50 nodes
model.add(Flatten()) # organizes layers
model.add(Dense(10, activation = 'softmax'))# output layer with nodes to match each category

# model.summary() # 

# Puts all the model feature together 
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy']) #loss='sparse_categorical_crossentropy'

# Now adding the training data in batch size of 1000
model.fit(X_train,y_train,batch_size = 1000)

# Model Score
m = model.evaluate(X_test,y_test)
m_eval = m[1]
m1 = m_eval*100
print('The model score is {} percent'.format(m1))

# Predicting 

yp = model.predict(X_test)
real = np.argmax(y_test[1])
print('The real is {}'.format(real))
predicted = np.argmax(yp[1])
print('The predicted is {}'.format(predicted))

"""
Your Turn...

My goal for you in this script is not for you to replicate but to understand

This is a more complex way of coding ANNs for more acceracy and precision compared to sklearn 
for bigger problems

Watch for more info: https://www.youtube.com/watch?v=aircAruvnKk&t=644s

# BONUS Change parameters 

Change the Activation functions to soft max
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Does the model improve?

"""
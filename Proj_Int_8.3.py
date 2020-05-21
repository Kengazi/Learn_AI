#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:20:28 2020

@author: kendalljohnson
"""

"CNN Keras"
"""
Week 8 - A base in using data science with python
8.3 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

Now That we have learned to use ANNs we will now focus an the AI technique called Convolutional Neural Networks(CNNs)
The only difference between CNNs and ANNs is that a CNN is an ANN with a convolutional layer

This convolutional layer takes into account the surronding pixels in an image.

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

"""
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.utils import to_categorical

""" 
This data is an 28x28 images of handwritten digits with changes in pixel intensities
"""

# Break into training and test data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# Normalizing pixels
X_test = X_test/ 255
X_train = X_train/ 255

# Categorizing
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshaping data
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Main 
model = Sequential()

#adding model layers

# Convolutional Layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Flatten 2D to hidden layer 1D
model.add(Flatten())

# Output of layers
model.add(Dense(10, activation = 'softmax'))

# model.summary() # Print all information

# Putting model layers together 
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy']) 

# Putting data in model
model.fit(X_train,y_train,batch_size = 1000)

# Model score
m = model.evaluate(X_test,y_test)
m_eval = m[1]
m1 = m_eval*100
print('The model score is {} percent'.format(m1))

# Predictions 
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

Watch for more CNN info: https://www.youtube.com/watch?v=FmpDIaiMIeA

# BONUS Change parameters 

Change the Activation functions to soft max
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Does the model improve?

# BONUS BONUS see if you can use keras to successfully create a Neural Network structure known as LeNet

"""
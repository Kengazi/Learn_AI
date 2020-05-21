#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:20:24 2020

@author: kendalljohnson
"""

"""
Week 8 - A base in using data science with python
8.1 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

Now that we have learned to use ANNs we will now focus an the statistical technique known as Confusion Matrix

The model score is a great Identify of how well your model has been made, but even a 100% model can make wrong predictions
A Confusion Matrix in the best way of showing how good your model really is by comparing your predicted output
with the real output and understanding were the mistakes were made. The middle diagnol is most important because 
it hold the correctly predicted outputs

"""
# Imports :: Modules used in code
import numpy as np
import matplotlib.pyplot as plt

# Sci-kit learns data analysis techniques 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

# Sci-kit learn's Artifical Neural Network Model or (Multi-Layer Perceptron)
from sklearn.neural_network import MLPClassifier

# Create an object to hold the digits dataset 
digits = datasets.load_digits()

""" 
This data is an 8x8 images of handwritten digits with changes in pixel intensities
"""

# Display example digits used in data
_, axes = plt.subplots(2, 2)

# Taking data from digits object
images_and_labels = list(zip(digits.images, digits.target))

# looping over number of presented images
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# Reshape data / Flatten
num = len(digits.images)
data = digits.images.reshape((num, -1))

# Artifical Neural Network Model or (Multi-Layer Perceptron)
model = MLPClassifier(activation='relu',solver='adam', alpha=1e-4,hidden_layer_sizes=(100,100),
                      random_state=5, learning_rate = 'adaptive',max_iter = 30000)

# Split data into 80% train and 20% test 
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False)

# Train model with training data
model.fit(X_train, y_train)

# Use model to predict test data
predicted = model.predict(X_test)

# Model score
M = model.score(X_test, y_test)
print("The score of this model is {:.4}%".format(M*100))

# Confusion Maxtrix 1 using slicing and matplotlib
images_and_predictions = list(zip(digits.images[num // 2:], predicted))
# Looping
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

# Classification of each class of data as a percetage of accuracy from sklearn's metric
print("Classification report for model %s:\n%s\n"
      % (model, metrics.classification_report(y_test, predicted)))

# Confusion Maxtrix 2 from sklearn's metric
disp = metrics.plot_confusion_matrix(model, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

# Plotting the Confusion matrix 
plt.show()


"""
Your turn...

What is your opinion of this model and its ability to pred? Specially for pred 8 and true 3

Create a confusion matrix for the Titanic data when x = class and y = lived

How many squares does your matrix have? 

How accurate is your model?

Try with SVM

How accurate is your model? Now

Which is better?

BONUS Try One other ML method, and Compare each model to which best works on the Titanic dataset

"""


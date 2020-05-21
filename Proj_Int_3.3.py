#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:57:15 2020

@author: kendalljohnson
"""

"""
Week 3 - A base in using data science with python

3.3 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with doing working with sklearn datasets and linear regression models.

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines

"""
# Title 
print('A base in using data science with python using pandas - guide')

# Imports
import numpy as np
np.random.seed(8)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso 
from sklearn.metrics import r2_score

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""

# Data import (the xlsx file has names attached)

df = pd.read_excel('Titanic.xlsx') 

# Data from titanic xlsx 
lived = df.values[:,1]              # 1 is survived 0 did not
clas = df.values[:,2]               # 1st class, 2nd, or 3rd
name = df.values[:,3]               # name of passenger "string"
gender = df.values[:,4]             # gender of passenger  "string"
age = df.values[:,5]                # age of passenger 
sib = df.values[:,6]                # number of brothers and sisters aboard
parent = df.values[:,7]             # number of parents / grandparents aboard
fare= df.values[:,8]                # how much their ticket cost


# Turning it into a dataframe
    
lived = np.array(lived)
lived_df = pd.DataFrame(lived) 

clas = np.array(clas)
clas_df = pd.DataFrame(clas)

name = np.array(name)
name_df = pd.DataFrame(name)

gender = np.array(gender)
gender_df = pd.DataFrame(gender)

age = np.array(age)
age_df = pd.DataFrame(age)

sib = np.array(sib)
sib_df = pd.DataFrame(sib)

parent = np.array(parent)
parent_df = pd.DataFrame(parent)

fare = np.array(fare)
fare_df = pd.DataFrame(fare)


"""
The training data is used to train the model to give it the ability to make assumetions of the test data
"""

print("Binary Logistic Regression")
# Seperate data into training and testing data using train_test_split making 80% of the data training and 20% test data

X_train, X_test, y_train, y_test = train_test_split(df[['parent']], df.lived, test_size = 0.1)

train_size = len(X_train)
test_size = len(X_test)

# Log Regression Model :: 

# Creating an object from import for use
model = LogisticRegression()

# Fitting the data in the model *** Important part ***
model.fit(X_train, y_train)

# Model Score *** Important value ***
m = (model.score(X_test,y_test)) * 100
print("The Model Score is {:.4}%".format(m))

# Using the model to preditions
y_pred = model.predict(X_test)

# R squared :: determiner of how good the prediction fits
r2 = (r2_score(y_test,y_pred)) * 100
print("The R2 Score is {:.4}".format(r2))

# Plotting Line of Best Fit based on model using Linear Model

plt.scatter(parent,lived,marker = '.',color = 'blue',label='data')
plt.plot(X_test,y_pred,color = 'red',label = "logistic line")  
plt.xlabel("Number of parents had")
plt.ylabel('Survived')
plt.grid()                                              # Plot grid lines 
plt.minorticks_on()                                     # adds small ticks on main axis
plt.legend()
plt.show()

# Multi-Logistic Regression
print("Multi-Logistic Regression")

# Seperate data into training and testing data using train_test_split making 80% of the data training and 20% test data
X1_train, X1_test, y1_train, y1_test = train_test_split(df[['parent','fare']], df.clas, test_size = 0.1)

# Log Regression Model :: 

# Creating an object from import for use
model = LogisticRegression()

# Fitting the data in the model *** Important part ***
model.fit(X1_train, y1_train)

# Model Score *** Important value ***
ms = (model.score(X1_test,y1_test)) * 100
print("The Model Score is {:.4}%".format(ms))

# Using the model to preditions
y1_pred = model.predict(X1_test)

# R squared :: determiner of how good the prediction fits
r_2 = (r2_score(y1_test,y1_pred)) * 100
print("The R2 Score is {:.4}".format(r_2))

# Plotting Line of Best Fit based on model using Linear Model

plt.scatter(y1_test,y1_pred,marker = '.',color = 'red', label='test vs pred')  
plt.xlabel("Real class")
plt.ylabel('Predicted class')
plt.grid()                                              # Plot grid lines 
plt.minorticks_on()                                     # adds small ticks on main axis
plt.legend()
plt.show()


"""
Your turn 

Explain why the graphs are poor but the model scores are our current best 
HINT:: it has something to do with continuos numbers and intergers

Create a Binary Logistic Regression model of age to pred survival

Create a Multi-Logistic Regression model of  siblings and parents to pred fare

Explain the graph and model scores of both

"""

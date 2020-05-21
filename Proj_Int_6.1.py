#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:13:40 2020

@author: kendalljohnson
"""


print('Using K-Fold \n')

"""
Week 6 - A base in using data science with python

6.1 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model to question them a show their true quility.

K-fold is basically finding the average model score of a data set with for choosen model, but it cycles through different
combos of test data and training data

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines
"""

# Title 
print('A base in using data science with python using pandas - guide')

# Imports :: modules used to create code

import pandas as pd                               # DataFrame Tool
import numpy as np                                # Basically a great calculator for now
from sklearn import linear_model                  # For our Linear Regression Model // Sklearn is a M.L. modules

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""

# Data import (the xlsx file has names attached)

df = pd.read_excel('Titanic.xlsx') 

# Data from titanic xlsx
numbers = df.values[:,0] 
lived = df.values[:,1]              # 1 is survived 0 did not
clas = df.values[:,2]               # 1st class, 2nd, or 3rd
name = df.values[:,3]               # name of passenger "string"
gender = df.values[:,4]             # gender of passenger  "string"
age = df.values[:,5]                # age of passenger 
sib = df.values[:,6]                # number of brothers and sisters aboard
parent = df.values[:,7]             # number of parents / grandparents aboard
fare = df.values[:,8]               # how much their ticket cost
num =len(age)
# Definition 

def Score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    S = model.score(X_test,y_test)*100
    return S

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

# Using Linear Regression

model = linear_model.LinearRegression()

# Using KFold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# K = 3
kf = KFold(n_splits = 3)

# Another Kfold method
KS  = StratifiedKFold(n_splits = 3)

models = []

for train_index, test_index in kf.split(numbers):
    X_train, X_test,y_train, y_test = age[train_index],age[test_index],age[train_index],age[test_index]
    train_size = len(train_index)
    test_size = len(test_index)
    # Reshape from (num,) to (num,1)
    X_train = X_train.reshape(train_size,1)
    y_train = y_train.reshape(train_size,1)
    X_test = X_test.reshape(test_size,1)
    y_test = y_test.reshape(test_size,1)
    S = Score(model,X_train, X_test,y_train, y_test)
    models.append(S)
    

avg = np.mean(models)
 
print("The average model score using KFold is {:}%".format(avg))   

"""
Your Turn

Change the n_splits in KFold.. What changes / improvements do you see?

Find the average model score for Logistic, SVM, and Randomforest for age

Find the average model score for Logistic, SVM, and Randomforest for gender

Find the average model score for Logistic, SVM, and Randomforest for fare

# BONUS use the StratifiedKFold method for these same data

"""
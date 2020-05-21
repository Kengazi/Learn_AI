#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:02:06 2020

@author: kendalljohnson
"""
# One - Hot - encoding
print('One - Hot - Encoding')

"""
Week 4 - A base in using data science with python

4.1 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with doing working with ML models like One - Hot - Encoding.

One - Hot - Encoding also you to turn string catagories to numbers to use in algorithms 

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines
"""
# Title 
print('A base in using data science with python using pandas - guide')

# Imports :: modules used to create code

import pandas as pd                               # DataFrame Tool
import numpy as np                                # Basically a great calculator for now
#import matplotlib.pyplot as plt                  # For 2-D Graphing
from sklearn import linear_model                  # For our Linear Regression Model // Sklearn is a M.L. modules

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
fare = df.values[:,8]                # how much their ticket cost


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

# Make Dummy varible that creates a binary representation of strings 
# We are going turn the strings "female" and "male" into 1 and 0

dum = pd.get_dummies(df.gender)                     # making the dummies using pandas
merged = pd.concat([df,dum],axis = 'columns')       # putting dataframes together
final = merged.drop(['male'],axis = 'columns')      # Removing 1 unneeded frames plus the last on the catigorical data frame

# Removing the strings, and lived because it is our y and also Unnamed because its not a needed
X = final.drop(['gender','name','lived',"Unnamed: 0"],axis='columns') # Removing independent variable 
y = final.lived                                     # independent variable alone

# Multi-Linear Reg used on onehot dummies and all numerical variables 
model = linear_model.LinearRegression()
model.fit(X,y)
F = model.score(X,y)
print('Models accuracy {:.4}'.format(F*100))

# [class(1-3),age(.42-80),sib(0-8),parent(0-6),fare(0-512),gender]
pred = model.predict([[3,20,0,1,20,1]])
pred = pred[0]
print("Prediction of survival is {:.4}".format(pred*100))

"""
Your Turn

Predict the Survival of

Matt = [2,42,1,0,57,0]

Ben = [3,14,3,2,8,0]

Leo = [3,23,0,0,5,0]

Rose = [1,21,1,3,150,1]

The Captian = [1,60,0,0,0,0]

Although we are now using all of the numerical data we still have a low model score 
In your opinon why?

Add the fake numerical data of how long it took for a passenger to get learn to boat was sinking in minutes.
Create a multi-linear regression model with that data

noise = np.abs(np.random.normal(0,1,887))
timeofsink = 20*noise

Then Predict the Survival of

Matt = [2,42,1,0,57,0,18]

Ben = [3,14,3,2,8,0,2]

Leo = [3,23,0,0,5,0,8]

Rose = [1,21,1,3,150,1,8]

The Captian = [1,60,0,0,0,0,0]


Add the fake string data of where a passenger are in the had life jackets "life_jacket"or not "no_life_jacket" 
Create a multi-linear regression model with that data

USE this to help 

jacket = []
for i in range(len(noise)): 
    trun = np.trunc(noise[i])
    if trun == 0:  
        jacket.append("no_life_jacket")
    else:
        jacket.append("life_jacket")

Use the one hot encoding to turn this is not 0s and 1s and perform a multi-linear regression model with that data

Then Predict the Survival of

Matt = [2,42,1,0,57,0,18,0]

Ben = [3,14,3,2,8,0,2,1]

Leo = [3,23,0,0,5,0,8,1]

Rose = [1,21,1,3,150,1,8,1]

The Captian = [1,60,0,0,0,0,0,0]

"""

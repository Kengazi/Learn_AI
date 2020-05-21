#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:09:41 2019

@author: kendalljohnson
"""


"""

Week 3 - A base in using data science with python

3.2 :: Using titanic data set for Machine Learning

The goal of this assignment is to get you comfortable with real datasets and muilt-linear regression models.

Scikit-learn is an amazing tool for machine learning that provides models in a few lines

"""
# Title 
print('A base in using data science with python using pandas - guide')

# Imports
import numpy as np
np.random.seed(2) # controls randomization not needed
import pandas as pd
# Sci-kit learn model
from sklearn.linear_model import LinearRegression

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""

# Data import (the xlsx file has names attached)
df = pd.read_excel('titanic.xlsx') 

# Data from titanic xlsx 
lived = df.values[:,0]              # 1 is survived 0 did not
clas = df.values[:,1]               # 1st class, 2nd, or 3rd
name = df.values[:,2]               # name of passenger "string"
gender = df.values[:,3]             # gender of passenger  "string"
age = df.values[:,4]                # age of passenger 
sib = df.values[:,5]                # number of brothers and sisters aboard
parent = df.values[:,6]             # number of parents / grandparents aboard
fare= df.values[:,7]                # how much their ticket cost


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

# Multi-Linear Regression Model :: 

# Creating an object from import for use
model = LinearRegression()

# Fitting the data in the model *** Important part ***
model.fit(df[['parent','sib']],df.age)

# Using the model to preditions with 
y_pred = np.abs(model.predict([[1,2]]))
y_pred = y_pred[0]

# Predicted age of Passenger
print("The predicted age of the passenger with {} parents and {} siblings on board is {:.4}".format(1,2,y_pred))

# Plotting 

# No graph with multi-varable "in this case"

"""
Your turn 

Create a multi-linear regression model with class and age to predict fare

Find 20 random predictions for fare

Check with similar fares in the data

Do you think your model worked well? How do you know

HINT Model score

BONUS Create this Multi-Linear Regression Model with seperated training values and test values
"""


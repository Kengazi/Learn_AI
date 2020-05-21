#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:13:46 2020

@author: kendalljohnson
"""

"Missing data"

"""
Week 6 - A base in using data science with python.

6.2 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model with missing data.

We will use different techniques in Pandas to fill the missing data that allows for a best prediction.

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines.
"""

# Title 
print('A base in using data science with python using pandas - guide')


# Imports :: modules used to create code

import pandas as pd                               # DataFrame Tool
import numpy as np                                # Basically a great calculator for now
np.random.seed(6)

# ML
from sklearn import linear_model                  # For our Linear Regression Model // Sklearn is a M.L. modules
from sklearn.model_selection import train_test_split

"""
This is fake data about the weather in June with varables like temperture (F), humidity (%), wind speed (mph),
and weather (as string). 
"""

# Data :: June excel file that has blanks in data please look at each df (Missing Data)

df = pd.read_excel('June.xlsx') # Has NaN (Missing Data)

new_df = df.fillna(0) # Fill NaN with 0

better_df = df.fillna({ # more accurate
        "Temp": 0, # Fills NaNs with zero
        "Humidity": 0,# Fills NaNs with zero
        "Weather": "Not Recorded"})# Fills NaNs with the String Not Recorded

foward_df = df.fillna(method="ffill") # fills NaNs with previous value

Back_df = df.fillna(method="bfill") # fills NaNs with next value

# Most Accurate and what I would like you to uses

b_df = df.fillna({ 
        "Weather": "Not Recorded"})
best_df = b_df.interpolate(method='linear') 

# Data from June xlsx 
Temp = best_df.values[:,0]              
Humidity = best_df.values[:,1]               # in percent
Wind = best_df.values[:,2] 
Weather = best_df.values[:,3]               # Weather as a "string"

# Turning it into a dataframe
Temp = np.array(Temp)
Temp_df = pd.DataFrame(Temp)

Humidity = np.array(Humidity)
Humidity_df = pd.DataFrame(Humidity)

Wind = np.array(Wind)
Wind_df = pd.DataFrame(Wind)

Weather = np.array(Weather)
Weather_df = pd.DataFrame(Weather)

# (One Hot Encoder) Make Dummy varible that creates a binary representation of strings 
# We are going turn the strings from weather to 1s and 0s

dum = pd.get_dummies(best_df.Weather)                     # making the dummies using pandas
merged = pd.concat([best_df,dum],axis = 'columns')       # putting dataframes together
final = merged.drop(['Not Recorded'],axis = 'columns')      # Removing 1 unneeded frames plus the last on the catigorical data frame
y = final.Sunny 

# Removing the strings, and lived because it is our y and also Unnamed because its not a needed
X = final.drop(['Weather','Sunny','Rain'],axis='columns') # Removing independent variable 

# X train and X test 
x_train, x_test,y_train, y_test = train_test_split(X,y,test_size = 0.2)

# Linear Reg used on onehot dummies and all numerical variables 
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
F = model.score(x_test,y_test)
print('Models accuracy {:.4}'.format(F*100))

# Predicting 

# [Temp(50-80), Humidity(15-100),Windspeed(2-25)]
pred = model.predict([[60,85,15]])
pred = np.abs(pred[0])

if pred > .5:
    print("Will likely be Sunny")  
    print("Pecertage is will be Sunny {:.4}%".format(pred))
else:
    print("Will likely Rain")
    print("Pecertage it will Rain {:.4}%".format(100-pred))
    
    
"""
Your Turn

Please look at all the changes of data set and comment your opinon of which is best to predict this data

Compare (model score) the method I used for missing data with the 2 other methods comment your opinon

In the orginal df file for lines 9,12,14,25 predict the weather

BONUS In the orginal df file for lines 4,13,16,18 predict the humidity

HINT you will need to change the to y = Humidity

"""

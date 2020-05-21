#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:13:44 2020

@author: kendalljohnson
"""

"Grad descent"

"""
Week 6 - A base in using data science with python

6.3 :: Using defined equations for Machine Learning.

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

We will introduce scipy optimize that allows us to make a like of best fit and compare it to the line we calculate with gradient descent.

Gradient descent is used in the next weeks introduction of Artifical Neural Networks

"""

# Title 
print('A base in using data science with python using pandas - guide')


# Imports 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""

# Data from titanic xlsx (the xlsx file has names attached)

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

# Definitions 

def Gradient_descent(x,y):
    m0 = b0 = 0                         # starting with zero slope and intercept
    iterations = 1000                   # How many times we mathmatically try to get closer to the true value of m and b (ment to be adjusted)
    n = x.size                          # Amount of x data
    learning_rate = 0.00001             # Learnin rate (ment to be adjusted)
    for i in range(iterations):
        y_pred = m0 * x + b0            # Predicted linear equations
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])           # Cost fuction
        dm = -(2/n) * sum(x*(y-y_pred)) # partially deriveved cost function with respected to m
        db = -(2/n) * sum(y-y_pred)     # partially deriveved cost function with respected to b
        m0 = m0 - dm * learning_rate    # Equation to update slope m
        b0 = b0 - db * learning_rate    # Equation to update intercept b
        print("m = {0}, b = {1}, cost = {2}, iterations = {3}".format(m0,b0,cost,i))
    pass
    return m0, b0

def Line(HR,m,b):
    y = m*HR + b
    return y

# Varibles and List
    
x = age
y = fare

# Scipy  Matrix :: easy way to get varibles in code using scipy.optimize
    
popt0,pcov0 = so.curve_fit(Line,x,y)

m0 = popt0[0]                           # slope value
b0 = popt0[1]                           # Intercept value

yy = m0 * x +b0

m, b = Gradient_descent(x,y)                  # Inacting the Gradient decent above
Y = Line(x,m,b)                          # For Gradient decent calc

# Plotting  
plt.scatter(x,y,color = "red",label = 'Real data', marker=".")                  # Plot Real Data
plt.title("Linear Model using Gradient descent")   # Plot title
plt.xlabel('Age of guest (Years)')                                     # Plot x axis name
plt.ylabel('Fare (pounds)')                                                   # Plot y axis label
plt.grid()                                                          # Plot grid lines 
plt.plot(x,Y,color = "blue",label = 'Grad line')                    # Plot Best fit line for Gradient Des   
plt.plot(x,yy,color = "green",label = 'Scipy line')                 # Plot Best fit line from Scipy                              
plt.legend()           
plt.minorticks_on()                                     # adds small ticks on main axis


"""
Your turn

The scipy line is the best way to get an accurate line and we see that the Grad and Scipy line are similar.
Up the interations of the Gradient descent model what do you see happen?
Up the learning rate of the Gradient descent model what do you see happen?

Try Gradient Descent on x = sib and y = parents

# BONUS used One Hot encoder to compare gender = x  to fare = y

"""
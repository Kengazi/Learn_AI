#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:02:25 2020

@author: kendalljohnson
"""

"""
***** first pip3 install pandas **************

Week 2 - A base in using data science with python

2.2 :: Upload Pandas DataFrames

The goal of this assignment is to get you comfortable with doing working with files in python using pandas.

Pandas is a very versitile module in python that allows you make, upload, and download files with ease.

"""

# Title 
print('A base in using data science with python using pandas - guide')

"""
This is heart data that has been recorded and we are going to analyse it.
We want to find the Heart rate of the indivual when exposed to a stimulas 
Then put it into an excel sheet with the calculate heart rates

"""

# Imports ::

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Raw data :: From measured activity 

df = pd.read_excel('Heart.xlsx')
numbers = df.values[:,0]            # list of number 
HR_A = df.values[:,1]               # Heart Rate in beats per minute
HR_K = df.values[:,2]               # Heart Rate in beats per minute
IBI_A = df.values[:,3]              # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  
IBI_K = df.values[:,4]              # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  
NN = df.values[:,5]   
SDNN = df.values[:,6]   
pNN20= df.values[:,7]   
pNN50= df.values[:,8] 
Stim_real = df.values[:,9]

# number of varables

num = len(numbers)

# Turning it into a dataframe
    
HR_A = np.array(HR_A)   
HR_A = pd.DataFrame(HR_A)

IBI_A = np.array(IBI_A)   
IBI_A = pd.DataFrame(IBI_A)

# To deteremine the NN    

NN = [] # list of number 
for i in range(num): 
    n = np.abs(IBI_K[i] - IBI_K[i-1])
    NN.append(n) 

# Turning it into a dataframe
    
NN = np.array(NN)   
NN = pd.DataFrame(NN)

# Varables from NN DataFrame the max, standard dev, mean, min, sum

# Max  
Max = NN.max()
Max = Max[0]

# Standard Devation (SDNN)
Std = NN.std() 
Std = Std[0]

# Minimum 
Min = NN.min()
Min = Min[0]

# Mean
Mean = NN.mean()
Mean = Mean[0]

# Sum
Sum = NN.sum()
Sum = Sum[0]
"""

The SDNN is very strong value to determine if a stimulas occur so we are going to use 
so we are going to test this theory by predicting our own list of stimulas occurance from the NN data 
if above our datas SDNN it will get a 1 and if below it will receive a 0

"""

# Turning it into a dataframe
    
SDNN = np.array(SDNN)   
SDNN = pd.DataFrame(SDNN)

# Using the SDNN to 
Stim_pred = []
for i in range(num): 
    if SDNN[0][i] > Std:
        Stim_pred.append(1)
    else:
        'The given SDNN values not bigger then calc SDNN'
        Stim_pred.append(0) 

# Turning it into a dataframe

Stim_pred = np.array(Stim_pred)   
Stim_pred = pd.DataFrame({'Stim':Stim_pred})

# Putting frames together (for only K and predict stim)
frames = [HR_A, IBI_A, NN, Stim_pred]
result = pd.concat(frames, axis=1)

# Bonus data representation
print(result.describe())

# Save as CSV
result.to_csv(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Heart_1.csv')

# Plotting 

# HR vs Pred Stimulus
plt.scatter(HR_A,Stim_pred,label = 'Training Data')     # Plot training Data
plt.title("HR vs Pred Stimulus")              # Plot title
plt.xlabel('HR(bpm)')                                   # Plot x axis name
plt.ylabel('Stimulus')                                  # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.minorticks_on()                                     # adds small ticks on main axis
plt.legend()                                            # Plot legend
plt.show()

# HR vs NN
plt.scatter(HR_A,NN,label = 'Training Data')     # Plot training Data
plt.title("HR vs NN")              # Plot title
plt.xlabel('HR(bpm)')                                       # Plot x axis name
plt.ylabel('NN')                                        # Plot y axis label
plt.minorticks_on()                                     # adds small ticks on main axis
plt.grid()                                              # Plot grid lines 
plt.legend()                                            # Plot legend
plt.show()

#Given SDNN and the one Calc SDNN
lists = np.linspace(0,num+1,num)
plt.scatter(lists,SDNN,label = 'Given Data')            # Plot Given Data
plt.scatter(7,Std,label = 'Calculated Data')            # Plot Calculated Data
plt.grid()                                              # adds grid lines
plt.xlabel("Inputs x")                                  # x label
plt.ylabel("Outputs y")                                 # y label
plt.title("Given SDNN and the one Calc SDNN ")          # Title of your graph
plt.minorticks_on()                                     # adds small ticks on main axis
plt.legend()                                            # Creates the legend make sure to put this after the plot data
plt.show()                                              # This plot show seperates the graph without it both would be on the same graph
#plt.savefig("Plot.png")                                # If you would like to save your plot


"""
Your turn

From looking at the graphs did this person has a stimulas, or stimilus-like events? 

Which graph gives best visual representation?

Find the mean, standard dev, max, and min of both HR, both IBI, NN, SDNN, pNN20, pNN50 of the Heart.xlsx data

Create a pandas dataframe and save as a csv file using the HR_K data and to predict the occurance of stimulas using the pNN20s which are the NNs over 20
and use scatter plot

Create a pandas dataframe and save as a csv file using the HR_K data and to predict the occurance of stimulas using the pNN50s which are the NNs over 50
and use scatter plot

HINT p stands for percent of NNs over 20 or 50 so, p_of_NNs_over_20 / total NN is the value you will need to use like I used Std

BONUS use different graphing techniques other then scatter plots, and what is you opinion of this graphing this data like this.
Is it a good visual representation?

"""



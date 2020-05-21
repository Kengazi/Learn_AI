#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:02:27 2020

@author: kendalljohnson
"""

"""
***** first pip3 install pandas **************

Week 2 - A base in using data science with python

2.3 :: Visual Analysis of Pandas DataFrames

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
HR_A_df = pd.DataFrame({'HR_A':HR_A}) 

HR_K = np.array(HR_K)
HR_K_df = pd.DataFrame({'HR_A':HR_K})

IBI_A = np.array(IBI_A)
IBI_A_df = pd.DataFrame({'IBI_A':IBI_A})

IBI_K = np.array(IBI_K)
IBI_K_df = pd.DataFrame({'IBI_A':IBI_K})

NN = np.array(NN)
NN_df = pd.DataFrame({'NN':NN})

SDNN = np.array(SDNN)
SDNN_df = pd.DataFrame({'SDNN':SDNN})

pNN20 = np.array(pNN20)
pNN20_df = pd.DataFrame({'pNN20':pNN20})

pNN50 = np.array(pNN50)
pNN50_df = pd.DataFrame({'pNN50':pNN50})

# Stimulus

Stim = np.array(Stim_real)
Stim_df = pd.DataFrame({'Stim':Stim})

# To deteremine the NN for K  

"""
NN is a very good determiner of a stimulas It represent time between beats, and the more time the more likely someone has a stimulas response

"""

NN_K = [] # list of number 
for i in range(num): 
    n = np.abs(IBI_K[i] - IBI_K[i-1])
    NN_K.append(n) 
    
# To deteremine the NN for A

NN_A = [] # list of number 
for i in range(num): 
    n = np.abs(IBI_A[i] - IBI_A[i-1])
    NN_A.append(n) 
    
# Putting frames together 

frame = [HR_A_df,HR_K_df, IBI_A_df, IBI_K_df, NN_df,SDNN_df,pNN20_df,pNN50_df,Stim_df]
result = pd.concat(frame, axis=1)
   
# HR vs NN
plt.scatter(HR_A,NN_A,color='red',label = 'Calc A')     # Plot training Data
plt.scatter(HR_K,NN_K,color='blue',label = 'Calc K')     # Plot training Data
plt.scatter(HR_A,NN,color='green',label = 'Given A')     # Plot training Data
plt.scatter(HR_K,NN,color='black',label = 'Given K')     # Plot training Data
plt.title("HR vs NN")              # Plot title
plt.xlabel('HR (bpm)')                                       # Plot x axis name
plt.ylabel('NN (ms)')                                        # Plot y axis label
plt.minorticks_on()                                     # adds small ticks on main axis
plt.grid()                                              # Plot grid lines 
plt.legend()                                            # Plot legend
plt.show()

# Save as excel file :: You need to change this to you saving areas 

result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Heart2.xlsx')

"""
Your turn 

Using the printed Graph
If you were the told that these are a random readings from 4 different heart moniters can you tell
which are stimulated events and which are not from lookin at the graph, and which heart reader is the highest quality 
and which is the lowest quality.

The red dots are the actually heart reader values. Do you think they were all accurate?

Now use the csv file HR4.csv with 460 HR values in Beats Per 10 Sec (not bpm) 
to make a graph of the NN above and ask the answer the same question as above.

"""
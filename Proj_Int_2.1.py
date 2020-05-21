#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:02:21 2020

@author: kendalljohnson
"""

"""
***** first pip3 install pandas **************


Week 2 - A base in using data science with python.

2.1 :: Create and Save Pandas DataFrames.

The goal of this assignment is to get you comfortable with doing working with files in python using pandas.

Pandas is a very versitile module in python that allows you make, upload, and download files with ease.

"""

# Title 
print('A base in using data science with python using pandas - guide')

"""
This is heart data that has been recorded and we are going to analyse it.
We want to find the Heart rate of the indivual when exposed to a stimulas 
Then put it into an excel sheet with the calculate heart rates

Heart Rate :: Beats of the heart minute (bpm)
IBI :: the inverse of the bpm in ms
NN :: NN = IBI[2nd] - IBI[1st]

"""

# Imports ::

import numpy as np
import pandas as pd

# Raw data :: From measured activity 

# IBI is interbeat intervals it is a property of the heart

IBI_A  = [738,656.95,697,718,651,849,663.7,666.69,673.77,680.43,716.76,630.4,702,660.19,706.09,702.1] # Real

IBI_K = [734,698,694,710,753,843,655,666,678,681,673,661,669,677,711,703]  # Real + noise and made to intergers 

# Value of the IBI

NN = [ 35., 22.,  6., 18., 47., 92., 181., 14., 11. ,5., 8., 18., 4., 7., 37., 9.]

# Value of the NN

SDNN = [73.6,47.1,36.65,43.05,209,51.7,52.6,42.48,34,47.75,54.8,42.42,48.74,50,40.85,45.98]

# Value of the NN

pNN20  = [64.7,67.5,62,58.9,100,73.9,65.7,66.13,52.6,64.5,75.36,73.33,83.3,69.7,64.1,61.6]

# Value of the NN

pNN50 = [47.06,32.4,25.9,22.3,100,39,35,24.2,18,25.8,37.7,43.3,30,39,25.6,29.3]

# If the heart was stimulated by an outside source at that time Stim occuring == 1

Stim = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0]


# Heart Rate(HR) = 60,000 / IBI
# IBI = 60,000 / HR

def HR(IBI):
    HR = 60000/IBI
    return HR

# making blank lists to fill with varables from the for loops
HR_A = [] 
HR_K = []

# For loop that uses HR def on each value of IBI

for i in range(len(IBI_A)):
    HR_a = HR(IBI_A[i])
    HR_k = HR(IBI_K[i])
    HR_A.append(HR_a)
    HR_K.append(HR_k)
    
# Making putting date in numpy array then turning numpy array to pandas data frames
# Pandas is a very effective way of storing and using data specially for machine learning    
# pd.DataFrame({label:data})
    
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

Stim = np.array(Stim)
Stim_df = pd.DataFrame({'Stim':Stim})
   
# Putting frames together 

frame = [HR_A_df,HR_K_df, IBI_A_df, IBI_K_df, NN_df,SDNN_df,pNN20_df,pNN50_df,Stim_df]
result = pd.concat(frame, axis=1)

# Save as excel file 

result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Heart.xlsx')

"""
Your turn 

Use this list of heart rates to create a list of IBIs by reverses the equation above 
Then make a pandas DataFrames of the HR,NR+noise, IBI, IBI+ noise, and Stimulus events 
Concatenate the frames 
Save as excel file name it heart_data.xlsx

HR = [84.7675, 75.9586, 78.4380, 72.2902 , 82.1259,71.4294, 81.8981, 89.2527 , 82.2481, 73.9046,88.0476, 84.9480, 54.9658, 72.0157, 66.8672,79.9352]
HR + noise
noise = np.random.normal(-5,5,HR_size) 
Stimulas = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

BONUS 

Use a for loop create a list of numbers(NN) by substracting an IBI by the previous IBI
Example np.abs(IBI[2] - IBI[1])
for actual and plus noise
Put in a list, then DataFrame with HR and IBI, and save as excel

"""

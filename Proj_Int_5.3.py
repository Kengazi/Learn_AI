#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:07:12 2020

@author: kendalljohnson
"""
#Using Kmeans to group the Clusters
"""
Week 5 - A base in using data science with python

5.3 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with doing working with ML models like K-means.

First unsupervised ML script. Used heavily for visual representations K is the number of groupings made on plot.

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines.
"""

# Title 
print('A base in using data science with python using pandas - guide')

# Imports ::
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

"""
This is heart data that has been recorded and we are going to analyse it.
We want to find the Heart rate of the indivual when exposed to a stimulas 
Then put it into an excel sheet with the calculate heart rates
"""

# Data 
df = pd.read_excel('P_1.xlsx')        # Heart Rate in beats per minute
time2 = df.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 

time2 = time2[:179]                   # Sliced to Certain place
IBI = df.values[:,4]
IBI = IBI[:179]                       # Sliced to Certain place
#Pat_size = HR.size


# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

T_df = pd.DataFrame(time2)
IBI_df = pd.DataFrame(IBI)

# Putting frames together 

frames = [T_df, IBI_df]
Com_df = pd.concat(frames, axis=1)

# Color Map size = K

#colmap = {1: 'r',2: 'g',3:'b'}
#colmap = {1: 'r',2: 'g',3:'b',4:'k'}
colmap = {1: 'r',2: 'g',3:'b',4:'k',5:'c'}
#colmap = {1: 'r',2: 'g',3:'b',4:'k',5:'c',6:'y'}
#colmap = {1: 'r',2: 'g',3:'b',4:'k',5:'c',6:'y',7:'w'}

# Machine Leanring

kmeans = KMeans(n_clusters=5) # The K choosen is 5
kmeans.fit(Com_df)

# Main Body
labels = kmeans.predict(Com_df)
centroids = kmeans.cluster_centers_

# Plotting
fig = plt.figure(figsize=(5,5))

# Colors Representations
colors = map(lambda x: colmap[x+1],labels)

colors1 = list(colors)

plt.scatter(T_df,IBI_df, color=colors1,alpha=0.5,edgecolor='k')

# Changing Cetroid(point in the middle of each k cluster) Mean
Cent = []
for idx, centroid in enumerate(centroids):
    Cent.append(centroid)                           
    plt.scatter(*centroid, color=colmap[idx+1])
    
plt.xlim(0,400) # Cut x axis to certien shape
plt.ylim(600,1000) # Cut y axis to certien shape
plt.title('K-Means of IBI(HR)')
plt.xlabel('Time intervals')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()

# Turned cetroid means to data frames 
Cent = pd.DataFrame(Cent)

# List of cetroid means
A = Cent.values[:,1]
print('IBI y values:',A)

"""
Your turn...

Change the K in the Kmeans and the number for color maps ... How does this changes the data

Uses the Titanic data with x = class y = fare and K = number of classes

What can you tell about this graph and do the groups seperate?

# BONUS use titanic's data with kids and parents to see if number of x = kids impacted the number of y = parents on board 

"""
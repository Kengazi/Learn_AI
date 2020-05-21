#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:18:06 2020

@author: kendalljohnson
"""

"""
***** first: sudo pip3 install pickle **************

Week 7 - A base in using data science with python
7.1 :: Introduction of Artifical Neural Networks for Machine Learning

Just as we used the early ML model we will use sklearn's Artifical Neural Network (ANN) Model 
aka Multi-Layer Perceptron(MLP)

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

Also we will learning to save these trained models using pickle
"""

# Title 
print('A base in using data science with python using pandas - guide')

# Imports 
import numpy as np 
import pandas as pd

# Sci-kit learn's Artifical Neural Network Model or (Multi-Layer Perceptron)
from sklearn.neural_network import MLPClassifier

# Save Model
import pickle

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""

# Data
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
num = len(age)


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

# Changing array shapes 
frame = [clas_df,age_df, sib_df, parent_df, fare_df] # Putting all data in array
inputs = pd.concat(frame, axis=1) # Concatinating it in a pandas data frame
inputs = inputs.to_numpy() # turning pandas DataFrame to numpy array
inputs = np.asarray(inputs,dtype=np.float64) # Changing values to float64
output = lived # Renaming
output = np.asarray(output,dtype=np.float64)# Changing values to float64

# Artifical Neural Network Model or (Multi-Layer Perceptron)
model = MLPClassifier(activation='logistic',solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(100,100,20),
                      random_state=5, learning_rate = 'adaptive',max_iter = 30000)
"""
Activation functions is logistic or sigmoid :: others relu,tanh.
Optimizer or solver is lbfgs or a Newton momemtum method :: others adam,sgd(gradient descent).
Hidden layer sizes are the shape of the ANN it start with the 5 inputs cat then 100,100,20 
then ends with 2 for 0 or 1 based on lived or died.
Maximum iteration is how many times the model is cacluated which is 3000 in this case it is also know as epochs
this is the part that takes time
"""

# Fitting Model like before
model.fit(inputs, output)

# Model Score like before
m = model.score(inputs,output) * 100
print("The model score is {:.4}%".format(m))

# Using Pickle to save models

# Saving model
ANN = 'ANN.pkl'
with open(ANN,'wb') as file:
    ANN_model = pickle.dumpd(ANN,file)

# Loading Model :: would normally be on a different script
ANN = 'ANN.pkl'
with open(ANN,'rb') as file:
    ANN_model = pickle.load(file)
    
# Predicting    
ANN_model.predict([])

"""
Your turn...

Clearly the most complex of Sklearns models what is your opinion of my favorite AI model

Change the Activation functions
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Does the model improve?

BONUS  use the one hot encoder on gender and add it as a catigory to the neural net

BONUS BONUS try to get an 85% model score with your changes
"""


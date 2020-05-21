#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:18:09 2020

@author: kendalljohnson
"""

"Pytorch ANN"


"""
***** first :sudo pip3 install torch **************
***** first :sudo pip3 install torchvision **************

Week 7 - A base in using data science with python

7.2 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

We are done with sklearn models and now will focus on Keras and Pytorch ANNs 

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

"""

# Title 
print('A base in using data science with python using Pytorch - guide')

# Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import datasets

# Data :: Creating a randomly corralated dataset
n = 500
X,y = datasets.make_circles(n_samples = n, random_state=120 , noise = .1, factor=.2)

# Storing data in a wat Pytorch can read
x_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(500,1))

# Works by creating a class that we will pull from. 
# Sklearn has classes that we use but they have already created them

# Our model 
class Model(nn.Module):
    
    def __init__(self,input_size,hidden, output_size): # Inputs
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden ) # Linear model
        self.linear2 = nn.Linear(hidden, output_size)# Linear model
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))          # Sigmoid function 
        x = torch.sigmoid(self.linear2(x))         # Sigmoid function 
        return x
    def predict(self,x): # Prediction Node
        pred = self.foward(x)
        if pred >= .5:
            return 1
        else:
            return 0
        
torch.manual_seed(2) # controls randomization like numpy.random.seed()

# Using Model
model = Model(2,4,1) 

# Looking at the parameters
print(list(model.parameters()))

# Another function of ANN called loss function (not present in sklearn models I used)
criterion = nn.BCELoss()

# Our optimizter, iteration(epochs) and learning rate
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)
epochs = 1001
losses = []

# Running model features :: like the Grad Descent from before
for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred,y_data)
    print('epoch:',i,"loss:",loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# Plotting how well the model ran
plt.plot(range(epochs),losses)
plt.title("Pytorch MLP/ ANN")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel('Loss')
    

"""
Your Turn...

My goal for you in this script is not for you to replicate but to understand

This is a more complex way of coding ANNs for more acceracy and precision compared to sklearn 
for bigger problems

Watch for more info: https://www.youtube.com/watch?v=aircAruvnKk&t=644s

# BONUS Change parameters 

Change the Activation functions
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Does the model improve?

"""
    
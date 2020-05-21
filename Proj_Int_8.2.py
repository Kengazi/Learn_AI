#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:20:26 2020

@author: kendalljohnson
"""

"CNN Pytorch"


"""
Week 8 - A base in using data science with python
8.2 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

Now That we have learned to use ANNs we will now focus an the AI technique called Convolutional Neural Networks(CNNs)
The only difference between CNNs and ANNs is that a CNN is an ANN with a convolutional layer

This convolutional layer takes into account the surronding pixels in an image.

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets,transforms

""" 
This data is an 28x28 images of handwritten digits with changes in pixel intensities
"""

# Data
transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

# Data MNIST digits 
training_data = datasets.MNIST(root="./data", train=True, download = True, transform=transform)
validation_data = datasets.MNIST(root="./data", train=False, download = True, transform=transform)

# Putting data into batches
training_loader = torch.utils.data.DataLoader(training_data, batch_size=100, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle = False)

# Making a class for model
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5,1) # Convolution layer(input,filters,)
        self.conv2 = nn.Conv2d(20,50,5,1) # Convolution layer(input,filters,)
        self.fc1 = nn.Linear(4*4*50,500) # Linear equation
        self.fc2 = nn.Linear(500,10) # Linear equation
    def forward(self,x):
        x = F.relu(self.conv1(x)) # Using relu function
        x = F.max_pool2d(x,2,2) # Using max pooling function
        x = F.relu(self.conv2(x)) # Using relu function
        x = F.max_pool2d(x,2,2) # Using max pooling function
        x = x.view(-1,4*4*50) # Put of the output
        x = F.relu(self.fc1(x)) # Using relu function
        x = self.fc2(x)
        return x
 
# Creating model as an object
model = Model()

# Loss Function
criterion = nn.CrossEntropyLoss()
# Obtimization Function
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)

# Number of interations to train model
epochs = 15

# History of model
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

# For loops that does iterations for losses to show improvement

for e in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
#  Regular running loss
    for inputs,labels in training_loader:
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _,preds = torch.max(outputs,1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
    else:
        with torch.no_grad():
            
#  Validation running loss
            for val_inputs,val_labels in validation_loader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs,val_labels)

                _,val_preds = torch.max(val_outputs,1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)
                
#  Epoch running loss    
    epoch_loss = running_loss/ len(training_loader)
    epoch_acc = running_corrects.float()/ len(training_loader)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    
#  Validation Epoch running loss 
    val_epoch_loss = running_loss/ len(validation_loader)
    val_epoch_acc = running_corrects.float()/ len(validation_loader)
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
    
# Printing Data
    print("epoch :", e)
    print("training loss: {:.4}, acc: {:.4}".format(epoch_loss, epoch_acc.item()))
    print("validation loss: {:.4}, validation acc: {:.4}".format(val_epoch_loss, val_epoch_acc.item()))
        
# Plotting
plt.plot(range(epochs),epoch_loss)
plt.title("Pytorch MLP/ CNN")
plt.grid()
plt.xlabel("Epochs")
plt.ylabel('Loss')

"""
Your Turn...

My goal for you in this script is not for you to replicate but to understand

This is a more complex way of coding ANNs for more acceracy and precision compared to sklearn 
for bigger problems

Watch for more CNN info: https://www.youtube.com/watch?v=FmpDIaiMIeA

# BONUS Change parameters 

Change the Activation functions to soft max
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Does the model improve?

"""
    
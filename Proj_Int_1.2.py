#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:57:22 2020

@author: kendalljohnson
"""

"""
Week 1 - A base in using data science with python

1.2 :: Numpy list and definitions.

The goal of this assignment are to get you comfortable with doing math in python with numpy.

NumPy is by far the most versitile module in python and least computationally expensive.

"""

# Title 
print('A base in using data science with python using numpy list, arrays, and definitions - guide')

# Import numpy

import numpy as np

# Varables

start_0 = 0
start_1 = 1
end = 10
num = 10

# Line / numpy array

line_0 = np.linspace(start_0,end,num+1)  # Very very useful
line_1 = np.linspace(start_1,end,num)  # Very very useful
# with linspace you can choose your start and end. 
# you can also choose how many varables that are in your line

zero=np.zeros(num) # list of all zeros
ones=np.ones(num)  # list of all ones 

# Slicing 
 
slice_0 = line_0[0]     # slicing to get values out of the list the [0] is the first value of the list
slice_1 = line_1[1]     # slicing to get values out of the list the [1] is the second value of the list


# Defining Functions of given equations examples
# this is an easier more dependable way of writting equations and functions

def a(x):               # The equation:  a(x) = x * 10
    F = x * 10
    return F

def F(A,x):             # The equation: F(x) = a*e^(x)
   F = A*np.exp(-x)
   return F

# Varables
   
A = 10

# Definitions can be used for more then Math

def answer(x):
    print("The answer is {:.4}".format(x))
    return x

# Answer to one varable slices from functions above
    
ans_0 = a(slice_0)
answer(ans_0)

ans_1 = F(A,slice_1)
answer(ans_1)

# Answers to lists with function used on them

answer_a = a(line_0)
answer_F = F(A,line_1)

# Using a for loop for definition 

for i in answer_a:
    answer(i)
    
# Define the same functions using the lambda function
    
func_a = lambda x:x*10                      # The equation:  a(x) = x * 10

func_F = lambda x:A*np.exp(-x)                # The equation: F(x) = a*e^(x)

# Answers to lambda functions
ans_lam_a = func_a(slice_0)
answer(ans_lam_a)

ans_lam_F = func_F(slice_1)
answer(ans_lam_F)

"""

 Your turn... please print the answers and round 

 Create a line using linspace from 10 to 20 that goes up by one 

 Create a line using linspace from 0 to 100 that goes up by one 

 Create a line using linspace from -2*pi to 2*pi with 20 points 

 Create a definition (def) for the pythagorean theory  c = sqrt(a**2 + b**2)

 now find the unknown side if a = 4 and b = 3; a = 20 and b = 10 

# BONUS :: create a definition using def and lambda for a shrinking cylinder 
  that has h = 10 meters and shrinks 5 times from a raduis of 100 meter to 10 meter
  what is the begining volume in m^3 and what is the end
  # hint use lists and a for loop

"""

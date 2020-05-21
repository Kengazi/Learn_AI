#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:57:24 2020

@author: kendalljohnson
"""

"""
Week 1 - A base in using data science with python

1.3 :: We will focus on more complex math and plotting with matplotlib

The goal of this assignment is to get you comfortable with doing math in python with numpy.

NumPy is by far the most versitile module in python and least computationally expensive.

"""

# Title 
print('\nA base in using data science with python - guide\n')

# Imports 
import numpy as np
import matplotlib.pyplot as plt

# Defining Functions of equations 

# A(x) = sin(x/100)
def A(x):
    A = np.sin(x/100)
    return A

# B(x) = e^(-x/200) * sin(x/100)
def B(x):
   B = np.exp(-x/200) * np.cos(x/100)
   return B 

# C(x) = abs(sin(1.5*ln(1-x^2)))
def C(x):
    C = np.abs(np.sin((1.5*np.log(1+x**2))))
    return C

# Varibles 
   
xa = np.linspace(0,1000,101)                                     # using linspace to make a set between 0 and 1000 with 101 digits
xb = np.linspace(0,1000,51)                                      # using linspace to make a set between 0 and 1000 with 51 digits
xc = np.linspace(0,10,1001)                                      # using linspace to make a set between 0 and 10 with 1001 digits

# Applying Functions

ya = A(xa) # Using functions defined above 
yb = B(xb) # Using functions defined above 
yc = C(xc) # Using functions defined above 

# Plotting 

# a plot
print('Graph A (Regular Plot)')
plt.plot(xa,ya,color='red',label='A(x)')                         # The actual plot data.. plot(x,y,color,label)
plt.title("Plot of x vs y Equ. F(x) = sin(x/100)")               # Title of your graph
plt.xlabel("Inputs x")                                           # x label
plt.ylabel("Outputs y")                                          # y label
plt.minorticks_on()                                              # adds small ticks on main axis
plt.legend()                                                     # Creates the legend make sure to put this after the plot data
plt.grid()                                                       # adds grid lines
plt.show()                                                       # This plot show seperates the graph without it both would be on the same graph

# b plot
print('Graph B (Scatter Plot)')
plt.scatter(xb,yb,color='blue',label='B(x)')                     # The actual plot data.. plot(x,y,color,label)
plt.title("Plot of x vs y Equ. F(x) = e^(-x/200) * sin(x/100)")  # Title of your graph
plt.xlabel("Inputs x")                                           # x label
plt.ylabel("Outputs y")                                          # y label
plt.minorticks_on()                                              # adds small ticks on main axis
plt.grid()                                                       # adds grid lines
plt.legend()                                                     # Creates the legend make sure to put this after the plot data
plt.show()                                                       # This plot show seperates the graph without it both would be on the same graph

# c plot
print('Graph C (Histogram)')
plt.hist(yc,color='green',label='C(x)')                          # The actual plot data.. plot(x,y,color,label)
plt.grid()                                                       # adds grid lines
plt.xlabel("Inputs x")                                           # x label
plt.ylabel("Outputs y")                                          # y label
plt.title("Experiment 2 Average mistake per min")                # Title of your graph
plt.minorticks_on()                                              # adds small ticks on main axis
plt.legend()                                                     # Creates the legend make sure to put this after the plot data
plt.show()                                                       # This plot show seperates the graph without it both would be on the same graph
#plt.savefig("Plot.png")                                         # If you would like to save your plot

"""
 Your turn now.. please use same graphing format I do
 
 Create a function and scatter plot for the eqaution for 
 
 y(x) = 2*x + e^(2*x) when x is -10 to 10 with 101 points
 
 F(a) = C * D * a * (v^2 * p)/2  C = .04, D = 1.005, v = 250, p = 0.00006, when a is 0.1 to 10 with 10001 points
 
 T_k(f) = (5/9)*(f-32) + 273.15 when f is -100 to 100 with 101 points
 
 SA(r) = pi*r**2 + pi*r*s ; when s = sgrt(a**2 + r**2); when r is 0.1 to 10 with 10001 points and a = 10
 
 k(E) = A*e^(-E/(R*T)) A = 1.85, R = 80, T = 1/(2*E), when E is 0 to 1 with 1001 points
 
 BONUS Which one decays to zero the fastest (put on same plot using regular plot)
 
 D = 1.234
 G = 2.54
 T = t/G
 t = 0 to 10 
 Da(t) = D*(1/2)^(t/T)
 Db(t) = D*e^(-t/T)
 Dc(t) = D*e^(G*t)
 
 
""" 
 
 
 
 
 
 
 
 
 
 
 
 

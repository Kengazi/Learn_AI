#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:44:12 2020

@author: kendalljohnson
"""
"""
POP QUIZ of all of the previous models used, because I am very impressed by your work.
"""
# Imports :: modules used to create code
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn import tree

# Datasets used
digits = load_digits()

# Splitting data
x_train, x_test,y_train, y_test = train_test_split(digits.data,digits.target,test_size = 0.2)

# Machine learning tech

# Machince Learning :: Linear Regression
Lin_Reg = LinearRegression()
Lin_Reg.fit(x_train,y_train)
print("The Linear Regression Score is {:.4}".format(Lin_Reg.score(x_test,y_test)))

# Machince Learning :: Logistic Regression
Log_Reg = LogisticRegression()
Log_Reg.fit(x_train,y_train)
print("The  Logistic Regression Score is {:.4}".format(Log_Reg.score(x_test,y_test)))

# Machince Learning :: Support Vector Machine 
SVM = SVC()
SVM.fit(x_train,y_train) #,criterion='4')
print("The  Support Vector Machine Score is {:.4}".format(SVM.score(x_test,y_test)))

# Machince Learning :: Decision Tree
Tree = tree.DecisionTreeClassifier()
Tree.fit(x_train,y_train) #,criterion='4')
print("The  Decision Tree Score is {:.4}".format(Tree.score(x_test,y_test)))
 

# Machince Learning :: Random Forest with 10 n_estimators
Forest = RandomForestClassifier(n_estimators=10)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {:.4} with 10 n_estimators".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 20 n_estimators
Forest = RandomForestClassifier(n_estimators=20)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {:.4} with 20 n_estimators".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 30 n_estimators
Forest = RandomForestClassifier(n_estimators=30)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {:.4} with 30 n_estimators".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 40 n_estimators
Forest = RandomForestClassifier(n_estimators=40)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {:.4} with 40 n_estimators".format(Forest.score(x_test,y_test)))

"""
This your pop quiz because you are great at this 

Part 1:
    Explain the data used.
    - How is it presented? Array, float, matrix, or etc
    - How much data is used / needed for good results ? HINT: training and testing percentages
    - What are the best 3 models and why in your opinon based on the data?
    - Validated the models in a way you know outside of model score. HINT: With lists of correct data predictions and their statistics 
    - What have you learned about the data from the models?
    - What have you learned about the models used?
    
Part 2:
    Apply the models above and do the same for this fashion dataset from the csv's from the link below.
    https://www.kaggle.com/zalando-research/fashionmnist

"""
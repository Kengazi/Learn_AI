#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:45:59 2020

@author: kendalljohnson
"""

"Opencv basic Image stuff"

"""
***** first: sudo pip3 install opencv ******* perferable 3.4 **************
***** And usable camera connected by usb directly *************************

###### I have only used this on the Pi and Nano with a Pi camera. Not my laptop or a webcam ###### 


### not the easiest to download I had download opencv from source, then use the make command that the build command


Week 9 - A base in using data science with python
9.1 :: Opencv is an easy to use platform that can perform visual recognition by using many ML and Stat techniques
We will start with simple technique then demonstraight the uses of CNN in object recognition using opencv

This is simply changing perameters of an image using the openCV module
"""
# Title 
print('A base in using data science with python using Open Computer Vision - guide')

# Imports
import matplotlib.pyplot as plt
import cv2

# definitions 

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

# Testing images
#img1 = ".jpg"
#img2 = ".jpg"
#img3 = ".jpg"
img = cv2.imread('Person.jpg',0)

# Show image in gray
plt.imshow(img,cmap='gray')

# Matrix inform
A = img.shape
h = A[0] # Height of image
w = A[1] # width of image

# Cutting image peice from full image shape 
#cut = img[380:500,550:w] # 

# Masks :: create all one color shape in image 
#mask = np.zeros(img.shape[:2],np.uint8)
#plt.imshow(mask)
#plt.imshow(cut,cmap='gray')
"""
 Template used to change property relationships of pixal on images.
 This is used to find irregularities or objects in the images 
 
"""
methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED',]

# Loop each method to find images
for m in methods:

# Make copy of image
    copy = img.copy()
# Apply Method
    method = eval(m)
#temp matching with 
    res = cv2.matchTemplate(copy,img,method) # replace img with cut
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)

# For a specific method that need to be reversed
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc# (x,y)
    else:
        top_left = max_loc
# Img properties
    height, width = img.shape # replace img with cut
    bottom_right = (top_left[0]+width,top_left[1]+height)

# Bounding boxes
    cv2.rectangle(img,top_left,bottom_right,(0,255,0),5)

# Plot heat map of pixal qualities
    plt.subplot(121)
    plt.imshow(res)
    plt.title("HEATMAP of Temp mapping")
    
# Plot Image
    plt.subplot(122)
    plt.imshow(img,cmap='gray')
    plt.title("Detection template")
    
# Title of method
    plt.suptitle(m)
    plt.show()
    print('\n')
    

"""
Your turn...

This is meant as a learning exercise for you understand about neccesary image proccessing

Which of the methods do you believes works best to make a heat map of the image

Choose your own 3 images img1,img2, and img3 to run in this script

Try and uncommenting the hash tags and see what the different properties do 

Try cutting the images to see differences in processing

BONUS Use the masks
"""
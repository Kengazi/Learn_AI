#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:46:04 2020

@author: kendalljohnson
"""

"Real time video masking"

"""
***** first: sudo pip3 install opencv ******* perferable 3.4 **************
***** And usable camera connected by usb directly *************************


Week 8 - A base in using data science with python

8.1 :: Opencv is an easy to use platform that can perform visual recognition by using many ML and Stat techniques.

We will start with simple technique then demonstraight the uses of CNN in object recognition using opencv.

This is simply changing perameters of an image using the openCV module
"""
# Title 
print('A base in using data science with python using Open Computer Vision - guide')

# Imports
import numpy as np
import cv2

# Open Computer Vision module
cap = cv2.VideoCapture(0)

# Varable
first = None

# Math kernals similar the Convolution layers
kernal1 = np.ones((1,1),np.uint8)
kernal2 = np.ones((2,2),np.uint8)
kernal3 = np.ones((3,3),np.uint8)
kernal4 = np.ones((4,4),np.uint8)

# Reading the video with while loop:: The video is essential images being taking past at certien rate and 
# we are proccessing each image coming in with a statitcal model

while(cap.isOpened()):
# The frame is each passing image, the ret in a true or false value that an image was found (usless to us)
    ret, frame = cap.read()
    
# Change image to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# Using a Guassian Blur to get rid of most noise
    gray = cv2.GaussianBlur(gray,(25,25),cv2.BORDER_DEFAULT) 

# store first image 
    if first is None: 
        first = gray
        continue
# Subtract new frame from previous
    delta_frame = cv2.absdiff(first,gray) # calc diff between frames
   
# Picking a range of pixal color on a thresh hold to show from the delta from    
    thresh_delta1 = cv2.threshold(delta_frame, 150, 255, cv2.THRESH_BINARY)[1]  #creates threshold of 30 if less then 30 its black

# Edits  Erode more black // dilate more black

# Erodinf Frame
    #erode1 = cv2.erode(thresh_delta1,kernal1, iterations=3)

# Dilating Frame
    dilate = cv2.dilate(thresh_delta1,kernal4,iterations=5)
    
# Another Guassian Blur
    Guass = cv2.GaussianBlur(dilate,(25,25),0) # convert gray scale to gaussian Blur
    
# adds borders   
    (_,cnts,_) = cv2.findContours(Guass.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
# removes noise and shadow
    for contour in cnts:
        if cv2.contourArea(contour) > 8000: # removes noise and shadow // better # bigger
           continue
        if cv2.contourArea(contour) < 10: # removes noise and shadow # Smaller
            continue
        
# Parameters of Bounding boxes
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

# Showing images just processed
    cv2.imshow('frame',frame) # No filter
    cv2.imshow('thresh',Guass) # with filter

# Press q to cancel
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera aperture
cap.release()

# Gets rid of previous (not delete) picture so we can bring up next
cv2.destroyAllWindows()


"""
Your turn...

This is meant as a learning exercise for you understand about neccesary video proccessing

Which of the methods do you believes works best for putting bounding boxes on an image

Try and uncommenting the hash tags and see what the different properties do 

BONUS Sub this line for line 32 for your own save video

cap = cv2.VideoCapture("/home/pi/Downloads/Your_own_saved_video")

Are you able to make object detections?
"""

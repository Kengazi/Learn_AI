#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:46:02 2020

@author: kendalljohnson
"""

"Opencv video stream"



"""
***** first: sudo pip3 install opencv ******* perferable 3.4 **************
***** And usable camera connected by usb directly *************************


Week 9 - A base in using data science with python
9.2 :: Opencv is an easy to use platform that can perform visual recognition by using many ML and Stat techniques
We will start with simple technique then demonstraight the uses of CNN in object recognition using opencv

Use Open CV to caputure constant video feed from camera and mask certein color object that allow the camera to put boxes around them

"""
# Title 
print('A base in using data science with python using Open Computer Vision - guide')

# Import
import numpy as np

# Open Computer Vision module 
import cv2 

"""
I have commented out the masks, so you can run the code without it. 
And with no masks it is simply a regular video stream.

When you and the mask you can point out things in the camera by their color

"""
# Creating video capture object
cap = cv2.VideoCapture(0)

# Creating while loops to continually cycle through frames
while True:

    # frame is essentially the image being loop
    _, frame = cap.read()
    # Convert BGR image to HSV
    #hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # the mask for the color red
    #low_red = np.array([161,155,84])
    #high_red = np.array([179,255,255])
    #red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    
    # the mask for the color yellow
    #bright_yellow = np.array([255,255,0])
    #low_yellow = np.array([200,200,50])
    #yellow_mask = cv2.inRange(hsv_frame, low_yellow,bright_yellow)
    
    # Finds controls on image
    #_, contours, _ = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # This line make have problems based on th opencv you choose  
    #contours = sorted(contours, key = lambda x: cv2.contourArea(x),reverse = True) # reverse from biggest to smallest
    # Create bounding boxes around colored items
    #for cnt in contours:
     #   (x,y,w,h) = cv2.boundingRect(cnt)
     #   cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
     #   break
    
    # Show images on screen
    cv2.imshow("Frame",frame) # image
    #cv2.imshow("mask", red_mask) # show only red
    #cv2.imshow("mask", yellow_mask) # show only color
    
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera aperture
cap.release()

# Gets rid of previous (not delete) picture so we can bring up next
cv2.destroyAllWindows()

"""
Your turn...

This is meant as a learning exercise for you understand about neccesary video proccessing

# Uncomment everything that is code

# Try using the red mask on a red object

# Try using the yellow mask on a yellow object

BONUS UCreate a blue mask to use on a blue object
"""
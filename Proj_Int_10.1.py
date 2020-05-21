#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:50:13 2020

@author: kendalljohnson
"""

"Haar casade face detect"

"""
********* first: sudo pip3 install opencv ******* perferable 3.4 **************
********* And usable camera connected by usb directly *************************


Week 10 - A base in using data science with python

10.1 :: Opencv is an easy to use platform that can perform visual recognition by using many ML and Stat techniques.

We will use what we have learned to create object detectors.

We will be using opencv to face detection using haar cascades. 
This is not really ML or AI but really a strong statistical technique that looks for face features, 
and when a face is found it looks for eye features.

Haar cascades creates a box then sum the pixel intensities.
the difference are then used to catagorize the sub sections of images
more feature testing the more accurate 
very fast the calculate features
"""



from __future__ import print_function # check that all print functions are the latest

# Imports
import numpy as np 
import cv2 # the open cv module 

# local modules:: files scripts in opencv
from video import create_capture # from the video module take a snap shot
from common import clock, draw_str # importing useful features example is the time each pic is taken

# Detection definition using the cascade data
def detect(img, cascade): # using the cascade to image detect 
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), 
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    # DetectMultiScale is an the CV function that analysis the pixels 
    if len(rects) == 0: # if no pic detected do nothing
        return []
    rects[:,2:] += rects[:,:2] # spliting and summing the picture pixels
    return rects

def draw_rects(img, rects, color): # Drawing the square around a face that is detected
    for x1, y1, x2, y2 in rects: # edges of the square
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) # creating the square to analyze 

if __name__ == '__main__': # import and run other modules 
    
    import sys, getopt
    # sys work with interpreter 
    # getopt is 
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])

#getopt is a C style parser that allows C like parsering sys.argv[1:] return the string

    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

# the classifiers from files:
    # Face
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    # Eyes
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")
    
# Creating objects of the classifiers
    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    
# Camera stream
    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

# Setting Camera to True for conituous viewing
    while True:
        ret, img = cam.read()

# Changing color to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
# Blurring a little
        gray = cv2.equalizeHist(gray)
        t = clock()
        
# Detect the images faces
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            
# Draw bounding boxes
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
                
# Time of detection
        dt = clock() - t
        
# Print info on detected image
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)
        
# Press q to end
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Get ride of previous frame to work on next
    cv2.destroyAllWindows()
    
    
"""
Your Turn...

Simply run the script and understand its components 

Is this an effective face detector?

Does it work realtime?

"""
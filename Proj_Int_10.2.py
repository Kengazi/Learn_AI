#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:50:15 2020

@author: kendalljohnson
"""

"OpenCV YOLO"

"""
********* first: sudo pip3 install opencv ******* perferable 3.4 **************
********* And usable camera connected by usb directly *************************
********** Download darknet from: https://pjreddie.com/darknet/yolo/***********

Week 10 - A base in using data science with python

10.2 :: Opencv is an easy to use platform that can perform visual recognition by using many ML and Stat techniques.

We will use what we have learned to create object detectors.

This is a model called YOLO (You only look once) with the coco dataset and it is an object detector.

It classifies objects from the coco dataset on your image and put a green bounding box around them.

"""
# Title 
print('A base in using data science with python using Open Computer Vision - guide')


# YOLO current fastest

# Imports
import numpy as np
import cv2

"""
# Importing the darknet files for the list of object that can be classified (yolov3.cfg) and 
# the weights saved from someone elses trained neural network
# and using cv2.dnn(deep neural network) to read the information
"""
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

# Read list of names to a file called classes
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

# Calling layers by their name; for example layer 1
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]- 1] for i in net.getUnconnectedOutLayers()]

# Picking colors
colors = np.random.uniform(0,255, size = (len(classes),3))

# Make a YOLO object for detecting
def yolo(img):
    
# incoming values from image 
    height, width, channels = img.shape
    
# Neural Network features 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    
# Blank lists
    confidences = []
    boxes = []
    class_ids = []
    
# For loops for detecting
    for out in outs:
        for detection in out: # confidences
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                
            # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
            # circle of detection
            #cv2.circle(img,(center_x,center_y,10,(0,255,0),2))
                x = int(center_x - w /2)
                y = int(center_y - h /2)
                
            # Bounding box parameters
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            #cv2.rectange(img, (x,y), (x+w, y + h, (0,255,0), 2))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
    
# Font choosen for boxes
    font = cv2.FONT_HERSHEY_PLAIN
    
# For loop to go through each detected feature and make a bounding box
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[i]
            print(label) # what is on the screen
            cv2.rectangle(img, (x,y), (x+w, y + h),color, 2)
            cv2.putText(img, label, (x,y + 30),font, 3,color, 3)
    return (x,y),(x+w,y+h),scores,label,img

# Start Camera
cap = cv2.VideoCapture(0)
cap.set(3,480) # shrink width
cap.set(4,320) # shrink height
a = 0 

# Make so there is constant importing
while True:
    _, frame = cap.read()
    
# Running definition and gettin back (corner points, score of model for pic, what the object is, and the image with boxes)
    (x,y),(x_t,y_t),scores, label, img =  yolo(frame)
    
# counting number of images
    a = a + 1
    print(a)

# Show camera capture image
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
# Release camera aperture
cap.release()

# Gets rid of previous (not delete) picture so we can bring up next
cv2.destroyAllWindows()


"""
Your Turn...

Simply run the script and understand its components 

Is this an effective object detector?

Does it work realtime?

"""
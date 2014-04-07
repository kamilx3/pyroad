# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/home/kamil/opencv/sign.png")
#img = cv2.bilateralFilter(img,30,175,175)
#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #COLOR_BGR2GRAY, COLOR_HLS2RGB, COLOR_RGB2HLS, COLOR_BGR2HLS

lower_blue = np.array([100,60,60])
upper_blue = np.array([140,255,255])
lower_white = np.array([0,0,220]) #Fix this
upper_white = np.array([360,255,255])
# Threshold the HSV image to get only blue colors
if False:
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
else:
    mask = cv2.inRange(hsv, lower_red, upper_red)
# Bitwise-AND mask and original image
#cv2.imshow('frame',img)
median = cv2.medianBlur(mask,5)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(median,kernel,iterations = 1)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 1) # not really neccessery, pretty slow in fact

##Contours finding
ret,thresh = cv2.threshold(closing,127,255,0) #unnecessary, does nothing
contours,hierarchy = cv2.findContours(thresh, 1, 1) #1, 5


res = cv2.bitwise_and(img,img, mask=closing)
#cnt = contours[0]
#M = cv2.moments(cnt)
#print M
#hull = cv2.convexHull(cnt)
#print len(contours)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if  cv2.contourArea(cnt)>0: #minimal sign size, aspect ratio
        cv2.drawContours(img, [cnt], 0, (0,255,0), 3) #change last to -1 to fill contour
cv2.imshow('img',img)
cv2.waitKey(0)

# Use this to classify sign type
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
# And hierarchy to find inner sign
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html
# And this to classify the specific sign
# http://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv

#        if True:
#            print cv2.arcLength(cnt, True)
#            hull = cv2.convexHull(cnt)
#            rect = cv2.minAreaRect(hull)
#            box = cv2.cv.BoxPoints(rect)
#            box = np.int0(box)
#            cv2.drawContours(img,[box],0,(0,0,255),2)
#        elif False:
#            (x,y),radius = cv2.minEnclosingCircle(cnt)
#            center = (int(x),int(y))
#            radius = int(radius)
#            cv2.circle(img,center,radius,(0,255,0),2)
#        else:
#            ellipse = cv2.fitEllipse(cnt)
#            cv2.ellipse(img,ellipse,(0,255,0),2)

#th3 = cv2.adaptiveThreshold(grey2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
#th2 = cv2.adaptiveThreshold(grey2,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,3)

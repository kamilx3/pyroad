# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("sign2.png")
#img = cv2.bilateralFilter(img,9,75,75) #slow
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #COLOR_BGR2GRAY, COLOR_HLS2RGB, COLOR_RGB2HLS, COLOR_BGR2HLS
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

low_w = np.array([0,0,200])
up_w = np.array([180,255,255])

kernel = np.ones((5,5),np.uint8)
maskw = cv2.inRange(hsv, low_w, up_w)
maskw = cv2.dilate(maskw,kernel,iterations = 1)

low_b = np.array([100, 160,0])  #make those values better
up_b = np.array([119,255,255])
low_r1 = np.array([0, 160, 0])
up_r1 = np.array([5, 255, 255])
low_r2 = np.array([170, 160, 0])
up_r2 = np.array([180, 255, 255])
low_y = np.array([16, 140, 0])
up_y = np.array([18, 255, 255])

for x in [[[low_b,up_b],["d"]],[[low_r1,up_r1],[low_r2,up_r2]],[[low_y,up_y]]]:
#for x in [[[low_b,up_b],["d"]]]:
    mask = cv2.inRange(hsv, x[0][0], x[0][1])
    for y in x[1:]:
        if y[0]=="d":
            mask = cv2.dilate(mask,kernel,iterations = 1)
        elif y[0]=="e":
            mask = cv2.erode(mask,kernel,iterations = 1)
        elif y[0]=="mo":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif y[0]=="mc":
            v2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 1)
        elif y[0]=="b":
            mask = cv2.medianBlur(mask,5)
        else:
            mask2 = cv2.inRange(hsv, y[0], y[1])
            mask = cv2.add(mask, mask2)
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 1) #1, 5
    #res = cv2.bitwise_and(img,img, mask=mask)
    for cnt in contours:
        cnt = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if  cv2.contourArea(cnt)>600 and aspect_ratio<1.5: #minimal size and width less than 1.5 of hight
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

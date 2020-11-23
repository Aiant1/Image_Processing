# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 00:59:13 2020

@author: Antika
"""

#1st HOMEWORK : 1. Prewitt edge detector: gradient filter  nonmaxima-suppression (NMS)

import cv2 as cv
import numpy as np
import math


i1, j1, i2, j2 = 0, 0, 0, 0
#nonmaxima-suppression (NMS)


def NMS(grad_mag, edge_orientation):

    rows, cols = grad_mag.shape
    supression_mat = np.copy(grad_mag)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            angle = float(edge_orientation[i, j])
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                i1, j1, i2, j2 = 1, 1, -1, -1
                
            elif 67.5 <= angle < 112.5:
                i1, j1, i2, j2 = 1, -1, -1, 1

            elif 22.5 <= angle < 67.5:
                i1, j1, i2, j2 = 1, 0, -1, 0

            else:
                i1, j1, i2, j2 = 0, -1, 0, -1

            C = grad_mag[i, j]
            A = grad_mag[i + i1, j + j1]
            B = grad_mag[i + i2, j + j2]
            
#If M(A) > M(C) or M(B) > M(C), discard pixel (x, y) by setting M(x, y) = 0
            if A > C or B > C:
                supression_mat[i, j] = 0

    return np.array(supression_mat).astype('uint8')

#Prewitt Gx 
GX = np.array([[1, 0,-1],
              [1, 0,-1],
              [1, 0,-1]])
    
#Prewitt Gx 

GY = np.array([[ 1, 1, 1],
              [ 0, 0, 0],
              [-1,-1,-1]])

image = cv.imread('C:\\Users\ASUS\\Desktop\\PYTHON\\antika_practice_1\\res\\Krishna.jpg')

grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
rows, cols, channels = image.shape

grad_mag = np.zeros((rows, cols))
edge_orientation = np.zeros((rows, cols))

print('The program is in process')
for x in range(1, rows - 1):
    for y in range(1, cols - 1):
        gx = np.sum(np.multiply(GX, grayscale_image[x-1:x+2, y-1:y+2])) / 3
        gy = np.sum(np.multiply(GY, grayscale_image[x-1:x+2, y-1:y+2])) / 3

        grad_mag[x, y] = math.sqrt(gx*gx + gy*gy)
        edge_orientation[x, y] = np.rad2deg(math.atan2(gx, gy))

grad_mag = np.array(grad_mag).astype('uint8')

#call NMS
nms = NMS(grad_mag, edge_orientation)

#Print the images
cv.imshow('origina_grayscale_image', grayscale_image)
cv.imshow('gradient_magnitude.', grad_mag)
cv.imshow('NMS', nms)

#Save the image

cv.imwrite('C:\\Users\ASUS\\Desktop\\PYTHON\\antika_practice_1\\output\\gradient_magnitude.jpg',grad_mag)
cv.imwrite('C:\\Users\ASUS\\Desktop\\PYTHON\\antika_practice_1\\output\\NMS.jpg',nms)


cv.waitKey(0)

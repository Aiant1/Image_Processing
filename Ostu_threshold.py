"""
Created on Mon Sep 26 00:00:13 2020

@author: Antika
"""


import numpy as np
import cv2
from sys import maxsize

h = [1]

def histogram(img):
    row, col = img.shape
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1
    x = np.arange(0, 256)

    return y


def regenerate_img(img, threshold):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= threshold:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y


def count_pixel(h):
    cnt = 0
    for i in range(0, len(h)):
        if h[i] > 0:
            cnt += h[i]
    return cnt


def wieght(s, e):
    w = 0
    for i in range(s, e):
        w += h[i]
    return w


def mean(s, e):
    m = 0
    w = wieght(s, e)
    for i in range(s, e):
        m += h[i] * i

    m = (m / float(w))if w != 0 else 0
    return m


def variance(s, e):
    v = 0
    m = mean(s, e)
    w = wieght(s, e)
    for i in range(s, e):
        v += ((i - m) ** 2) * h[i]

    v = (v / m) if m != 0 else 0
    return v


def threshold(h):
    cnt = count_pixel(h)
    min_v2w = maxsize
    optimal_t = 0
    for i in range(0, 256):
        vb = variance(0, i)
        wb = wieght(0, i) / float(cnt)

        vf = variance(i, len(h))
        wf = wieght(i, len(h)) / float(cnt)

        V2w = wb * (vb) + wf * (vf)

        if min_v2w > V2w:
            min_v2w = V2w
            optimal_t = i

    return optimal_t


image = cv2.imread('res/aluminium.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h = histogram(img)
optimal_t = threshold(h)
print("OPTIMAL THRESHOLD:", optimal_t)

res = regenerate_img(img, optimal_t)
cv2.imwrite("C:\\Users\\ASUS\\Desktop\\PYTHON\\antika_practice_2\\res\\otsu_aluminium.jpg", res)
cv2.imshow('orig', image)
cv2.imshow('otsu', res)

cv2.waitKey(0)
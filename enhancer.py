# Student: Kaue Ueda Silveira
# NUSP: 7987498

import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# 1. Parameter Input
filename = str(input())
gamma = float(input())
alpha = float(input())
beta = float(input())
show_flag = int(input())

# 2. Read the input image
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# 3. Aply transformations
imgL = imLog(img)
imgG = imGamma(img)
imgH = imEqualHist(img)
imgS = imSharp(img)

cv2.destroyAllWindows()

def imLog(imgf)
    c = 255 / math.log(1 + imgf.max())

    imgg = cv2.log(np.float32(imgf)+1)
    imgg = cv2.multiply(imgg, c)

    return imgg


def imGamma(imgf, gamma):
    imgg = cv2.pow(np.float32(imgf), gamma)

    return imgg    


def imEqualHist(imgf):
    m, n = imgf.shape[:2]

    factor = 255/(m*n)
    histogram = imHistogram(imgf)
    imgg = imgf

    for mi in range(m):
        for ni in range(n):
        imgg[mi][ni] = factor*histogram[imgf[mi][ni]]

    return imgg

def imSharp(imgf, a, b):
    # Sharpening array
    # 0.05   0.1   0.05
    # 0.1   0.4   0.1
    # 0.05   0.1   0.05
    w = np.array([5, 10, 5, 10, 40, 10, 5, 10, 5])/100
    imgb = cv2.filter2D(imgf, -1, w)

    imgbf = cv2.subtract(imgb, imgf)
    imgbeta = cv2.multiply(imgbf, b)
    imgalpha = cv2.multiply(imgf, a)
    imgg = cv2.add(imgalpha, imgbeta)

    return imgg

def imHistogram(imgf):
    hist = [0]*256

    # grab image F and transforms it into a list of pixels
    imgl = list(imgf.ravel())

    # counts the number os pixels of each color (gray shade)
    for i, val in enumerate(imgl):
        hist[imgl[i]] += 1

    return hist

def showHistogram(imgf):
    plt.hist(imgf, 256, [0, 256])
    plt.show()
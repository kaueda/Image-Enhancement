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

if()

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
    


def imSharp(imgf):
    pass


def imHistogram(imgf):
    # numpy way
    # hist, bins = np.histogram(imgf.ravel(), 256, [0, 256])
    # return hist

    #my way
    hist = []
    imgl = list(imgf.ravel())
    for i in range(256):
        hist.append(imgl.count(i))

    return hist

def showHistogram(h):
    plt.hist(h, 256, [0, 256])
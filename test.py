import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def imHistogram(imgf):
    hist = [0]*256

    # grab image F and transforms it into a list of pixels
    imgl = list(imgf.ravel())

    # counts the number os pixels of each color (gray shade)
    for i, val in enumerate(imgl):
        hist[imgl[i]] += 1

    return hist

def showHistogram(hist):
    plt.hist(hist, 256, [0, 256])
    plt.show()

img = cv2.imread("nap.jpg", cv2.IMREAD_GRAYSCALE)
showHistogram(imHistogram(img))
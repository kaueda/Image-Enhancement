import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def imHistogram(imgf):
    # numpy way
    hist, bins = np.histogram(imgf.ravel(), 256, [0, 256])
    return hist

    #my way
    # hist = []
    # imgl = list(imgf.ravel())
    # for i in range(256):
    #     hist.append(imgl.count(i))

    # return hist

def showHistogram(hist):
    plt.hist(hist, bins='auto')
    plt.show()


img = cv2.imread("nap.jpg", cv2.IMREAD_GRAYSCALE)
showHistogram(imHistogram(img))
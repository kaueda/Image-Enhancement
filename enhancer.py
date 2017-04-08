# Student: Kaue Ueda Silveira
# NUSP: 7987498

import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

##
## Enhancing Functions
##

def imLog(imgf):
    c = 255 / math.log(1 + imgf.max())

    imgg = cv2.log(np.float32(imgf)+1)
    imgg = cv2.multiply(imgg, c)
    imgg = cv2.convertScaleAbs(imgg)
    imgg = cv2.normalize(imgg, imgg, 0, 255, cv2.NORM_MINMAX)

    return imgg

def imGamma(imgf, gamma):
    # First we normalize the data to be between [0, 1]
    imgg = cv2.normalize(imgf, imgf, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Then we aplly the gamma transformation
    imgg = cv2.pow(imgg, gamma)
    # Now we simply bring back the image to [0, 255] scale
    imgg = cv2.multiply(imgg, 255)
    imgg = cv2.convertScaleAbs(imgg)

    return imgg

def imEqualHist(imgf):
    m, n = imgf.shape[:2]

    # Factor (L-1)/(MxN)
    factor = 255/(m*n)

    # Cumulative distribution of each gray value
    cdf = np.array(imHistogram(imgf)).cumsum()
    # Minimum non-zero value of cdf
    cdfmin = cdf[np.nonzero(cdf)].min()

    # Apply equalisation
    imgg = np.float32(imgf)
    for mi in range(m):
        for ni in range(n):
            imgg[mi][ni] = (cdf[imgf[mi][ni]] - cdfmin)*factor

    imgg = cv2.convertScaleAbs(imgg)

    return imgg

def imSharp(imgf, a, b):
    # Sharpening array
    # 0.05   0.1   0.05
    # 0.1   0.4   0.1
    # 0.05   0.1   0.05
    w = np.array([5, 10, 5, 10, 40, 10, 5, 10, 5])/100
    imgb = cv2.filter2D(np.float32(imgf), -1, w)

    # Subtract b(x, y) and f(x, y)
    imgbf = cv2.subtract(imgb, np.float32(imgf))
    # Multiply by beta
    imgbeta = cv2.multiply(imgbf, b)
    # Multiply f(x, y) by alpha
    imgalpha = cv2.multiply(np.float32(imgf), a)
    # Add both parts
    imgg = cv2.add(imgalpha, imgbeta)
    imgg = cv2.convertScaleAbs(imgg)

    return imgg

def imHistogram(imgf):
    hist = [0]*256

    # grab image F and transforms it into a list of pixels
    imgl = list(imgf.ravel())

    # counts the number os pixels of each color (gray shade)
    for i, val in enumerate(imgl):
        hist[imgl[i]] += 1

    return hist

def showHistogram(imgf, islist=False):
    if(islist):
        for img, lbl in imgf:
            plt.hist(img.ravel(), 256, [0, 256], label=lbl)

        plt.legend()

    else:
        plt.hist(imgf.ravel(), 256, [0, 256])
    
    plt.show()

def rmsd(imgf, imgg):
    m, n = imgf.shape[:2]
    res = 0.0

    for x in range(m):
        for y in range(n):
            res += math.pow(float(imgf[x][y]) - float(imgg[x][y]), 2)

    res /= n*m

    return math.sqrt(res)

# 1. Parameter Input
filename = str(input())
gamma = float(input())
alpha = float(input())
beta = float(input())
show_flag = int(input())

# 2. Read the input image
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("could not load image.")
    exit()

# 3. Aply transformations
imgL = imLog(img)
imgG = imGamma(img, gamma)
imgH = imEqualHist(img)
imgS = imSharp(img, alpha, beta)

# 4. Show images (depending on flag)
if (show_flag == 1):
    cv2.imshow("Original Image", img)
    cv2.imshow("Logarithm Image", imgL)
    cv2.imshow("Gamma Image", imgG)
    cv2.imshow("Histogram Equal Image", imgH)
    cv2.imshow("Sharpening Image", imgS)

    # 5. Show histogram
    showHistogram([(img, "original"),
                   (imgL, "log"), 
                   (imgG, "gamma"),
                   (imgH, "histogram"), 
                   (imgS, "sharpening")]
    , islist=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 6. Output
print("RMSD")
print("L=" + str(rmsd(img, imgL)))
print("G=" + str(rmsd(img, imgG)))
print("H=" + str(rmsd(img, imgH)))
print("S=" + str(rmsd(img, imgS)))
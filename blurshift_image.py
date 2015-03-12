import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
import scipy.io
from matplotlib import pyplot as plt

def gauss2D(shape,sigma):
    #creating a gaussian kernal
    m = round(sigma*3.5)
##    print m
    n = round(sigma*3.5)    
    x = np.ogrid[-int(m):int(m)+1]
    y = np.ogrid[-int(n):int(n)+1]
    h = np.exp( -(x*x+y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
##    if sumh != 0:
##        h /= sumh
    return h


def blurshift(img_blur,sigma1,alpha,rhoi,delta_xi,delta_yi):
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    rows,cols = img_blur.shape
    #the std deviation of the guassian kernal 
##    sigma1 = sigma0+alpha*(int(rhoi))

##    img_blur = cv2.GaussianBlur(img_blur,(int(round(sigma1)),int(round(sigma1))),0)
    filter_gaussian = gauss2D((rows,cols),sigma1)

    # the shift in x and y direction for DOG response
##    delta_xi = -1*int(rhoi)*np.cos(int(phii))
##    delta_yi = -1*int(rhoi)*np.sin(int(phii))
    #the const value should lie in between -3 and 3
    x1 = sigma1
    y1 = sigma1
    rows1,cols1 =img_blur.shape
    M = np.float32([[1,0,delta_xi+x1],[0,1,delta_yi+y1]])
    #translating the response
    Dogresp_trans = cv2.warpAffine(img_blur,M,(cols1,rows1))
    #mulitplying the coefficients of gaussian and translated DOG response
##    Dog_blur_shifted = cv2.multiply(Dogresp_trans.astype(float),filter_gaussian)
    dilation = cv2.dilate(Dogresp_trans,filter_gaussian,iterations = 1)
##    dilated = cv2.dilate(dilation,filter_gaussian.transpose(),iterations = 1)
    return dilation

img_blur = cv2.imread('A4.jpg')
img_blur = cv2.GaussianBlur(img_blur,(3,3),0)
output = blurshift(img_blur,0.733,0.1167,4,0,0)
####cv2.namedWindow('img_transalte',1)
####cv2.imshow('img_translate',output)
####cv2.waitKey(0)
cv2.imwrite('output4.jpg',output)

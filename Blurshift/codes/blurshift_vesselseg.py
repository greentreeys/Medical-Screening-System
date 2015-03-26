import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
import scipy.io
from matplotlib import pyplot as plt

def gauss2D(shape,sigma):
    #creating a gaussian kernal
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def blurshift(img_blur,sigma0,rhoi,phii):
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    rows,cols = img_blur.shape
    alpha = 1
    #the std deviation of the guassian kernal 
    sigma1 = int(sigma0)+alpha*(int(rhoi))

    ##img_blur = cv2.GaussianBlur(img,(sigma1,sigma1),0)
    filter_gaussian = gauss2D((rows,cols),sigma1)
    ##plt.imshow(filter_gaussian, cmap=plt.get_cmap('jet'), interpolation='nearest')
    ##plt.colorbar()
    ##plt.show()

    # the shift in x and y direction for DOG response
    delta_xi = -1*int(rhoi)*np.cos(int(phii))
    delta_yi = -1*int(rhoi)*np.sin(int(phii))
    x1 = sigma1
    y1 = sigma1
    rows1,cols1 =img_blur.shape
    M = np.float32([[1,0,delta_xi+x1],[0,1,delta_yi+y1]])
    #translating the response 
    Dogresp_trans = cv2.warpAffine(img_blur,M,(cols1,rows1))
    #mulitplying the coefficients of gaussian and translated DOG response
    Dog_blur_shifted = cv2.multiply(Dogresp_trans.astype(float),filter_gaussian)
    return Dog_blur_shifted

img_blur = cv2.imread('tempblur.jpg')
output = blurshift(img_blur,2,1,1)
cv2.namedWindow('img_transalte',1)
cv2.imshow('img_translate',output)
cv2.waitKey(0)





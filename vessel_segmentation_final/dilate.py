import numpy as np
import cv2
import cv2.cv as cv
##from PIL import Image
##import scipy
##import scipy.io
##from matplotlib import pyplot as plt

def dilate(image,kernel):
    kernelSize,c = kernel.shape
##    print kernelSize
    kCenterX = kernelSize/2
    kCenterY = kernelSize/2
    rows,cols = image.shape
    output = np.zeros(image.shape,dtype = float)
    image_new = np.zeros((image.shape[0]+kernelSize,image.shape[1]+kernelSize),dtype=float)
    image_new[kernelSize/2:rows+kernelSize/2,kernelSize/2:cols+kernelSize/2]=image[:,:]
    r,c = image_new.shape
    for i in range(kernelSize/2,r-2*(kernelSize+1)):
        for j in range(kernelSize/2,c-2*(kernelSize+1)):
            mx = 0.0      
            tmp = image_new[i-kCenterX+3:kernelSize+i-kCenterX+3,j-kCenterY+3:kernelSize+j-kCenterY+3]*kernel
            mx = np.amax(tmp)
            output[i][j]=mx
    mi = np.amin(output)
    ma = np.amax(output)
    output = np.divide((np.subtract(output,mi)),(ma-mi))
    return output

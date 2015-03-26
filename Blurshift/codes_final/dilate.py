import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
import scipy.io
from matplotlib import pyplot as plt

def dilate(image,kernel):
    kernelSize,c = kernel.shape
    kCenterX = kernelSize/2
    kCenterY = kernelSize/2
    rows,cols = image.shape
    output = np.zeros(image.shape,dtype = float)
    for i in range(0,rows):
        for j in range(0,cols):
            sum = 0
            mx = 0.0
            tmp = 0.0
            for m in range(0,kernelSize):
                for n in range(0,kernelSize):
                    rowIndex = i + m - kCenterY
                    colIndex = j + n - kCenterX
                    if(rowIndex >=0 and rowIndex < rows and colIndex >= 0 and colIndex < cols):
                        tmp = image[rowIndex][colIndex]*kernel[m][n]
                        if (tmp>mx):
                            mx=tmp
            output[i][j]=mx
            
    return output

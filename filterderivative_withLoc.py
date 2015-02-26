import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
#from matplotlib import pyplot as plt
""" [L Lw Lvv Lvw Lww]= filterderivative(im,scale)
im= input single channel image
Scale at which derivation computation required: default: [1 2 4 8 16];
L Lw Lvv Lvw Lww : return derivative vector: dim == as per the given scales
feats= 25 vector per pixel only belongs to the disk region.
loc= contained respective location of the pixel which can be later used to display assigned cluster
[r,g,b]hist= are the histogram of OD pixels with total of 25 bins (10 bins are mapped to one[255-> 25bins])"""

im = cv2.imread('sample.jpg') #Read the original image
mask = cv2.imread('mask.jpg') #Read mask image
[m, n, dim] = im.shape

if dim > 0:
	im = cv2.split(im)[1]#extracting the green channel 
	im = im.astype(float)
	print "filterderivative: <Extracted Green Channel from the color image >  "
else :
	im = im.astype(float)
green = im
#disp ('Curvature space Lvv is computed.. ')

scale = raw_input("scale: ")
if len(scale) == 0:
	std = [1,2,4,8,16] #Number of scale

else :
	std = scale

[xx, yy] = np.where(np.all(mask == 1, axis=-1))
feats = np.zeros((len(xx),25), dtype = np.int)

[tmp, num] = [1,len(std)]
epsilon = 1e-2

rows,cols = im.shape
Lvv = np.zeros((rows,cols,len(std)), dtype = np.int)
Lvw = np.zeros((rows,cols,len(std)), dtype = np.int)
Lww = np.zeros((rows,cols,len(std)), dtype = np.int)
pos = 1

for k in range(num):
	sigma = std[k]
#To create Gaussian filter similiar to the one used in gaussgradient function
	with np.errstate(invalid='ignore'):
		halfsize = np.ceil(sigma*(np.sqrt(-2*(np.log(np.sqrt(2*np.pi)))*sigma*epsilon)))
	sze = 2*halfsize + 1
	#45Mat gauss = createGaussianFilter(sze, sigma)
	
#	cv::mulTransposed(gaussianKernel,gaussianKernel,false)
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

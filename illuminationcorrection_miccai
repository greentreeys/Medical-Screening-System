import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
import scipy.io
from matplotlib import pyplot as plt

img = cv2.imread('eye.png',1) #Read the original image
img_temp = img
sz_ori_r, sz_ori_c, sz_ori_d = img.shape
sz_ori_d= 0

img = img.astype(float)
mask = cv2.imread('f.tif',0)#read the mask
mask = cv2.resize(mask,(1024,1024))#resizing the mask
#convert the 255 value to 1
mi = np.amin(mask)
ma = np.amax(mask)
#normalise the image between 0 and 1
mask = np.divide((np.subtract(mask,mi)),(ma-mi))
img = cv2.resize(img, (1024,1024))#resizing image
img = cv2.split(img)[1]#extracting the green channel 
minimum = np.amin(img)
maximum = np.amax(img)
#normalise the image between 0 and 1
img = np.divide((np.subtract(img,minimum)),(maximum-minimum))

block_r = 512 #block height
block_c = 512 #block width
sz_r ,sz_c = img.shape
size = (sz_r/block_r , sz_c/block_c)
mean_sub = np.zeros((sz_r/block_r , sz_c/block_c), dtype=np.double) # Preallocate mean and std_dev matrices
std_sub = np.zeros((sz_r/block_r , sz_c/block_c), dtype=np.double)

for counter_row in range(sz_r/block_r):
    for counter_col in range(sz_r/block_c):
        # Next four lines compute coordinates for defining the patch
        top_left_r = block_r *(counter_row) 
        height = block_r * (counter_row+1)
        top_left_c = block_c * (counter_col)
        width = block_c * (counter_col+1)
        mask_patch = mask[top_left_r : height , top_left_c : width]
        
        #finding the non-zero elements in mask_patch
        I = mask_patch >0
        
        temp = img[top_left_r : height , top_left_c : width].astype(float)
        temp = temp[mask_patch>0]
        
        if (np.sum(I)==0):
            mean_sub[counter_row, counter_col] = 0
            std_sub[counter_row, counter_col] = 1
        else:
            means = np.mean(temp)
            stddevs = np.sqrt(np.var(temp)) #Computation of mean and std
            mean_sub[counter_row, counter_col] = means
            std_sub[counter_row, counter_col] = stddevs

row_img , col_img = img.shape
mean_full = np.zeros((row_img,col_img),dtype = np.double)
cv2.resize(mean_sub,(col_img,row_img),mean_full,interpolation = cv2.INTER_CUBIC) # Interpolate the mean_dev
mask = mask.astype(float)
cv2.multiply(mean_full,mask,mean_full)#multiplication by mask
std_full = cv2.resize(std_sub,(col_img,row_img)) #interpolate stddev
cv2.multiply(std_full,mask,std_full)#multiplication by mask
img_sub = cv2.subtract(img, mean_full)
pcm_dist = cv2.divide(img_sub, std_full)

pcm_dist = np.abs(pcm_dist)   # Compute the absolute value of the distance as it should be positive
cv2.multiply(pcm_dist,mask,pcm_dist)
pcm_dist[pcm_dist<1]= 1 #choose the threshold globally as 1
pcm_dist[pcm_dist!=1] = 0

block_r = 512
block_c = 512
for counter_row in range(sz_r/block_r):
    for counter_col in range(sz_r/block_c):
        top_left_r = block_r *(counter_row) 
        height = block_r * (counter_row+1)
        top_left_c = block_c * (counter_col)
        width = block_c * (counter_col+1)
        
        temp = img[top_left_r : height , top_left_c : width].astype(float)
        temp_pcm = pcm_dist[top_left_r : height , top_left_c : width]
        temp_mask = mask[top_left_r : height, top_left_c : width]
        I = temp_pcm >0 # Find the non-zero pixels in the background
        I2 = temp_mask >0
        I3 = cv2.bitwise_and(I.astype(int),I2.astype(int))
        temp = temp[I3==1]
        if (np.sum(I)==0):
            mean_sub[counter_row, counter_col] = 0
            std_sub[counter_row, counter_col] = 1
        else:
            means = np.mean(temp)  #Compute the mean and std_dev for the corresponding intensitities
            stddevs = np.sqrt(np.var(temp))#Computation of mean and std
            mean_sub[counter_row, counter_col] = means
            std_sub[counter_row, counter_col] = stddevs
      
row_img , col_img = img.shape
mean_full = np.zeros((row_img,col_img),dtype = np.double)
cv2.resize(mean_sub,(col_img,row_img),mean_full,interpolation = cv2.INTER_CUBIC) # Interpolate the mean_dev
mask = mask.astype(float)
cv2.multiply(mean_full,mask,mean_full)#multiplication by mask

std_full = cv2.resize(std_sub,(col_img,row_img)) #interpolate stddev
cv2.multiply(std_full,mask,std_full)#multiplication by mask)
img_sub = cv2.subtract(img, mean_full)
corrected = cv2.divide(img_sub, std_full)
corrected =  cv2.resize(corrected,(sz_ori_c,sz_ori_r))
mask =  cv2.resize(mask,(sz_ori_c,sz_ori_r))

min_crct = np.amin(corrected)
max_crct = np.amax(corrected)
##normalise the image between 0 and 1
corrected = np.divide((np.subtract(corrected,min_crct)),(max_crct-min_crct))
corrected[mask==0]=0
min_crct = np.amin(corrected)
max_crct = np.amax(corrected)
##normalise the image between 0 and 1
corrected = np.divide((np.subtract(corrected,min_crct)),(max_crct-min_crct))
mean_full =  cv2.resize(mean_full,(sz_ori_c,sz_ori_r)) #Interpolate to find the luminosity
std_full =  cv2.resize(std_full,(sz_ori_c,sz_ori_r))  #Interpolate to find the contrast

plt.subplot(221),plt.imshow(cv2.resize(img,(sz_ori_c,sz_ori_r)),'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([]) 
plt.subplot(222),plt.imshow(corrected,'gray'),plt.title('Corrected')
plt.xticks([]), plt.yticks([]) 
plt.subplot(223),plt.imshow(mean_full,'gray'),plt.title('luminosity')
plt.xticks([]), plt.yticks([]) 
plt.subplot(224),plt.imshow(std_full,'gray'),plt.title('contrast')
plt.xticks([]), plt.yticks([]) 
plt.show()
cv2.waitKey(0)

img_temp = cv2.imread('eye.png')

img_temp = img_temp.astype(float)
b,g,r = cv2.split(img_temp)
img2 = cv2.merge([r,g,b])
img_s = cv2.max(b,g)
v = cv2.max(img_s,r)
out = np.zeros(img_temp.shape,dtype=float)
B= cv2.divide((cv2.multiply(corrected,b)),v)
G= cv2.divide((cv2.multiply(corrected,g)),v)
R= cv2.divide((cv2.multiply(corrected,r)),v)
out = cv2.merge([R,G,B])
min_crct = np.amin(img2)
max_crct = np.amax(img2)
##normalise the image between 0 and 1
img2 = np.divide((np.subtract(img2,min_crct)),(max_crct-min_crct))

#subplot of original and corrected images
plt.subplot(121),plt.imshow(img2),plt.title('Original')
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(out),plt.title('Corrected')
plt.xticks([]), plt.yticks([]) 
plt.show()
cv2.waitKey(0)

cv2.destroyAllWindows()

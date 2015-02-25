import numpy as np
import cv2
import cv2.cv as cv
from PIL import Image
import scipy
import scipy.io
from matplotlib import pyplot as plt

##Example:
##Inputs:
##"Enter the number of sample points (16/36/64) : 64 "
##img = cv2.imread('eye.png',1)
##mask = cv2.imread('f.tif',0)


#the sample points can be 16 or 36 or 64
spoints = raw_input("Enter the number of sample points (16/36/64): ")


img = cv2.imread('eye.jpg',1) #Read the original image
img_temp = img
sz_ori_r, sz_ori_c, sz_ori_d = img.shape
sz_ori_d= 0

img = img.astype(float)
#mask = cv2.imread('f.tif',0)#read the mask
b,g,r = cv2.split(img)
mask = np.zeros(b.shape,dtype=np.uint8)
I = (b > 10)
mask[I] = 255
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

#depecnding upon the smapling points setting the window size
#and the number of iterations needed

if int(spoints) == 16:
    no_iter = 2
    r2 = 300
elif(int(spoints) == 36):
    no_iter = 3
    r2 = 250
else:
    no_iter=4
    r2=200

mean_sub = np.zeros((2*no_iter , 2*no_iter), dtype=np.double) # Preallocate mean and std_dev matrices
std_sub = np.zeros((2*no_iter , 2*no_iter), dtype=np.double)

cx = 512
cy = 512

sum_iter = (no_iter)*(no_iter+1)/2
l1 = np.pi/2 #initial window size(angle)
l=l1
k=0 #initial angle
r1 = 0 #initial radius
temp_r = r2

for iter in range(no_iter):
    if iter == no_iter-1:
        r2 = 512
    k=0 #for every iteration k should start from 0
    for i in range(4*(2*(iter)+1)):
        mask1 = np.zeros((sz_r,sz_c), dtype=np.double)
        #creating mask to calculate mean and std dev
        for x in range(1024):
            for y in range(1024):
                if(np.sqrt((cx-x)**2 + (cy-y)**2)<r2 and(np.sqrt((cx-x)**2 + (cy-y)**2))>r1 and (np.arctan2(cy-y,cx-x)+np.pi) < (k+l) and (np.arctan2(cy-y,cx-x)+np.pi)>k):
                    if(mask[x,y]==1):
                        mask1[x,y]=1
        I=mask1>0    
        temp = cv2.multiply(img,mask1).astype(float)
        temp = temp[mask1>0]
        mean_subx = np.ceil((iter+1)*np.cos(k+(l/2)))
        mean_suby = np.ceil((iter+1)*np.sin(k+(l/2)))
        
        if (np.sum(I)==0):
            mean_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = 0
            std_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = 1
        else:
            means = np.mean(temp)
            stddevs = np.sqrt(np.var(temp)) #Computation of mean and std
            mean_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = means
            std_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = stddevs
        k=k+l
    l=l1/(2*(iter+1)+1)
    r1=r2
    if (int(spoints) ==  16):
        r2=r2+200
    elif (int(spoints) == 36):
        r2 = r2 +150 - 50*(iter-1)
    else:
        r2 = r2 +150 - 50*(iter-1)
        
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

cx = 512
cy = 512

sum_iter = (no_iter)*(no_iter+1)/2
l1 = np.pi/2 #initial window size(angle)
l=l1
k=0 #initial angle
r1 = 0 #initial radius
r2 = temp_r 

for iter in range(no_iter):
    if iter == no_iter-1:
        r2 = 512
    k=0 #for every iteration k should start from 0
    for i in range(4*(2*(iter)+1)):
        mask1 = np.zeros((sz_r,sz_c), dtype=np.double)
        #creating mask to calculate mean and std dev
        for x in range(1024):
            for y in range(1024):
                if(np.sqrt((cx-x)**2 + (cy-y)**2)<r2 and(np.sqrt((cx-x)**2 + (cy-y)**2))>r1 and (np.arctan2(cy-y,cx-x)+np.pi) < (k+l) and (np.arctan2(cy-y,cx-x)+np.pi)>k):
                    if(mask[x,y]==1):
                        mask1[x,y]=1
        
        pcm_temp = cv2.multiply(pcm_dist,mask1).astype(float)
        temp = cv2.multiply(img,mask1).astype(float)
        I = pcm_temp >0
        I2 = temp > 0
        # take the pixels which are there bith in pcm_temp and temp
        I3 = cv2.bitwise_and(I.astype(int),I2.astype(int))
        temp = temp[I3==1]
        mean_subx = np.ceil((iter+1)*np.cos(k+(l/2)))
        mean_suby = np.ceil((iter+1)*np.sin(k+(l/2)))
        
        if (np.sum(I)==0):
            mean_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = 0
            std_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = 1
        else:
            means = np.mean(temp)
            stddevs = np.sqrt(np.var(temp)) #Computation of mean and std
            mean_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = means
            std_sub[(no_iter + mean_subx)-1,(no_iter + mean_suby)-1] = stddevs
        k=k+l
    l=l1/(2*(iter+1)+1)
    r1=r2
    if (int(spoints) ==  16):
        r2=r2+200
    elif (int(spoints) == 36):
        r2 = r2 +150 - 50*(iter-1)
    else:
        r2 = r2 +150 - 50*(iter-1)
        
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
    
#img_temp = cv2.imread('eye.png')

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

##subplot of the original and the corrected images
plt.subplot(121),plt.imshow(img2),plt.title('Original')
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(out),plt.title('Corrected')
plt.xticks([]), plt.yticks([]) 
plt.show()
cv2.waitKey(0)

cv2.destroyAllWindows()




    
    



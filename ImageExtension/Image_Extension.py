import numpy as np
import cv2


###########part1
# Load an color image in grayscale and mask  
img = cv2.imread('E:/cvit/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image040.png',1);
mask=cv2.imread('E:/cvit/diaretdb1_v_1_1/resources/images/ddb1_fundusmask/fmask.tif',0);


############part2
#morphological operations to find boundary of foreground or background of fundus image

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)); #structural element of size 20*20
mask = cv2.erode(mask,kernel,iterations = 1); #erosion of image

kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); #structural element of size 3*3
bound_mask=cv2.erode(mask,kernel1); #erosion of image

boundary=mask & (~bound_mask); #extracted boundary image


############part3

non_zero=cv2.findNonZero(boundary);#finding non zero pixel locations in the boundary
row,row_r,row_c=non_zero.shape;
#rearraging non_zero matrix to form as x*2 matrix 

n_z = np.zeros((row,row_c),np.float32); 

for i in xrange(row):
    for j in xrange(row_c):
        n_z[i,j]=non_zero[i][0,j]
        
#morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)); #structural element
mask_eroded = cv2.erode(mask,kernel,iterations = 1); #erosion of image

non_zero_1=cv2.findNonZero(~mask)
row_1,row_r_1,row_c_1=non_zero_1.shape;
n_z1= np.zeros((row_1,row_c_1),np.float32);


for i in xrange(row_1):
    for j in xrange(row_c_1):
        n_z1[i,j]=non_zero_1[i][0,j]



# knn search for nearest neighbours
responses = np.zeros((row,1),np.float32) ##creating labels

for i in range(row):
    responses[i,0]=i;
knn = cv2.KNearest()
knn.train(n_z,responses)
ret, results, neigh ,dist = knn.find_nearest(n_z1, 1)





nt=neigh.astype(int);
n_zt=n_z.astype(int);
n_z1t=n_z1.astype(int);




###looping
try:
    for i in xrange(row_1):
        img[n_z1t[i,1],n_z1t[i,0],:]=img[((2*n_zt[(nt[i,0]),1])-n_z1t[i,1]),((2*n_zt[(nt[i,0]),0])-n_z1t[i,0]),:];
except:
    print("There is a subscript problem")
    print("the size of image is ",img.shape);
    
cv2.imwrite('F:/messigray.png',img)

cv2.namedWindow('w', cv2.WINDOW_NORMAL)
cv2.imshow("w",img);


     
     
    




















########knn match




#cv2.imshow('image',boundary);


cv2.waitKey(0)
cv2.imwrite('F:/image.jpg',img)
cv2.destroyAllWindows()

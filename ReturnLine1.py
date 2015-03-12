# This code returns a horizontal line passing through the OD
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

origimg = cv2.imread('2.png')
origimg = origimg[:,:,2]
imgvess = cv2.imread('bp4.jpg')
temp =cv2.imread('imgt4.jpg')
x,y = imgvess.shape[:2]
origimg = cv2.resize(origimg,  (x,y))
#sh = cv2.imread('fig.jpg')
minn = 100000000.00000000

Xaxis = []
Yaxis = []
for i in range(0,(int)(x/8)-1):
	Xaxis.insert(i,0)
	Yaxis.insert(i,0)

for i in range((x/8), (x-x/8)):
	ind = np.where(imgvess[int(i)] == 255)
	ind = ind[0]
	#print 1
	on = ind.shape
	#Xaxis[i] = i
	Xaxis.insert(int(i),int(i))
	#Yaxis[i]=1
	Yaxis.insert(int(i),1)
	#Xaxis[i] = i
	#Yaxis[i] = 1
	ind = np.where(imgvess[0:i] == 255)
	ind = ind[0]
	inz = np.where(imgvess[0:i] == 0)
	inz = inz[0]
	c1 = ind.shape
	#print c1
	z1 = inz.shape
	#print z1
	dc1 = c1[0]/(float)(z1[0]+c1[0])
	#print dc1
	ind = np.where(imgvess[i:x] == 255)
	ind = ind[0]
	inz = np.where(imgvess[0:i] == 0)
	inz = inz[0]
	c2 = ind.shape
	z1 = inz.shape
	dc2 = c2[0]/(float)(z1[0]+c2[0])
	#print dc2
	diffones = abs(c2[0]-c1[0])
	diffden = abs(dc2-dc1)
	differ = diffden
	if (minn > differ):
		minn = differ
		xi = i
	
#imgvess[xi-10:xi+10][:] = 255

print (x/8)
print xi
cv2.imwrite('result.jpg', imgvess)


#crop original image
crop_img = origimg[xi-50:xi+250,:]
crop_img = crop_img

cx, cy = crop_img.shape[:2]


#taking horizontal and vertical projections
horizontaledge = cv2.Sobel(crop_img, cv2.CV_64F, 0, 1, ksize = 5)
verticaledge = cv2.Sobel(crop_img, cv2.CV_64F, 1,0, ksize = 5)
feature1 = abs((horizontaledge - verticaledge))
feature2 = (horizontaledge + verticaledge)
#for i in range(0,cx-1):
#	for j in range(0,cy - 1):
#		feature1[i][j] = feature1[i][j]/crop_img[i][j]
#		feature2[i][j] = feature2[i][j]/crop_img[i][j]

     

tot=sum(sum(crop_img))
hp = sum(feature1)
vp = sum(feature2)

plt.figure(1)
plt.plot(np.log(hp))
plt.show()
plt.figure(2)
plt.plot(np.log(vp))
plt.show()
implot=plt.imshow(crop_img)
plt.scatter(max(hp),max(vp))
#cv2.circle(crop_img,(int(xcor),int(ycor)),10,(0,0,255),-1)
cv2.imshow('result',crop_img)
cv2.waitKey(0)



























'''


hp = np.zeros((1,cy), np.uint8)
vp = np.zeros((1,cx), np.uint8)

maxh = np.where(hp==max(hp))
print maxh
x1 = maxh[0].shape[0]

#plt.figure(1)
#plt.plot(hp)
#plt.show()

maxv = np.where(vp == max(vp))
print maxv
y1 = maxv[0].shape[0]
#plt.figure(2)
#plt.plot(vp)
#plt.show()
#crop_img[maxv[0][0]][maxh[0][0]] = 0

cv2.circle(crop_img, (maxv[0][0],maxh[0][0]), 10, [0,255,0], -1)

cv2.imshow('cropped', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow('cropped', crop_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


'''

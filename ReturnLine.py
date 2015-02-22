# This code returns a horizontal line passing through the OD
import numpy as np
import cv2
from PIL import Image

imgvess = cv2.imread('bp.jpg')
temp =cv2.imread('imgt.jpg')
x,y = imgvess.shape[:2]
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
	
imgvess[xi-10:xi+10][:] = 255

print (x/8)
print xi
cv2.imshow('res',imgvess)
cv2.waitKey(0)
cv2.destroyAllWindows()





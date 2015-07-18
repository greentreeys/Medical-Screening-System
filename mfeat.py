# SVM Training for macula

import numpy as np
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import os

mypath = './extended'
images = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]

featvec = []
klabels = []

for w in images:
	print 'Image:',w
	mac_mask = cv2.imread('./mac/' + w) # Reads macula mask
	height, width, depth = mac_mask.shape
	height=(int)(height/2)
        width=(int)(width/2)
	print 'width:',width
	print 'height:', height
	mac_mask = cv2.cvtColor(mac_mask, cv2.COLOR_BGR2GRAY)
	mac_mask=cv2.resize(mac_mask,(height, width))
	m = np.where(mac_mask>0) # Marks the macula region
	print np.shape(m[0])
	orig = cv2.imread('./extended/' + w) # Reads original image
	orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	
	ves_mask = cv2.imread('./vessels/' + w) # Reads vessel mask
	ves_mask = cv2.cvtColor(ves_mask, cv2.COLOR_BGR2GRAY)
	ves_mask=cv2.resize(ves_mask,(height, width))
	print 'size of ves_mask:', ves_mask.shape	
	print 'size of mac_mask:', mac_mask.shape
# Findng the centre of the OD mask  - (X,Y) ----------------------------
	od_mask = cv2.imread('./od/' + w)    # Reads od mask
	od_mask = cv2.cvtColor(od_mask, cv2.COLOR_BGR2GRAY)
	od_mask=cv2.resize(od_mask,(height, width))
	orig=cv2.resize(orig,(height, width))
	contours, hierarchy = cv2.findContours(od_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	maxm = 0
	ind = 0
	for i in range(len(contours)):
		if maxm<len(contours[i]):
			maxm = len(contours[i])
			ind = i
	cnt= contours[ind]
	cv2.drawContours(od_mask,[cnt],ind,255,6)
	x,y,w,h = cv2.boundingRect(cnt)
	img = cv2.rectangle(od_mask,(x,y),(x+w,y+h),255,2)
	img = cv2.circle(od_mask, (x + int(w/2), y+ int(h/2)), 2, 255,2)
	
	X = x + int(w/2)
	Y = y + int(h/2)

# positive samples
	rands = np.random.choice(np.shape(m[0])[0], 30, replace=False)
	print 'random indices chosen:', rands
	for i in rands:
		xind = m[0][i]
		yind = m[1][i]
		xdist = np.abs(X - xind)
		ydist = np.abs(Y - yind)
# line joining the two centres
		den_above = 0
		slope = 0
		cnt_above = 0
		cnt_below = 0
		den_below = 0
		density = 0
		if (xdist!=0):
			slope = float(Y - yind)/(X - xind)
			for u in range(0,width):
				for v in range(0,height):
					if ((v - slope*u + slope*X - Y)>0):
						den_above = den_above + ves_mask[u][v]/255
						cnt_above = cnt_above + 1
					elif ((v - slope*u + slope*X - Y)<0):
						den_below = den_below + ves_mask[u][v]/255
						cnt_below = cnt_below + 1
					else:
						den_below = 1
					 	den_above = den_below
					
			dabove = float(den_above)/cnt_above
			dbelow = float(den_below)/cnt_below
			density = float(dabove)/dbelow
		else:
			density = 0

		fv = [xdist, ydist,density]
		k = [1]
		featvec.append(fv)
		klabels.append(k)
# Negative samples
	origcopy=orig
	print 'size of orig:' , origcopy.shape
	print 'size of od_mask: ', od_mask.shape
	outp = cv2.bitwise_and(origcopy,origcopy,mask = od_mask)
	negs = cv2.bitwise_and(outp, outp,mask = mac_mask)
	n = np.where(negs == 0) # These are indices of points that are non- macula, non- od
	rrnds = np.random.choice(np.shape(n[0])[0], 30, replace = False) # Randomly pick 30 negative samples
	for j in rrnds:
		xind = n[0][j]
		yind = n[1][j]
		xdist = np.abs(X - xind)
		ydist = np.abs(Y - yind)

		den_above = 0
		den_below = 0
		cnt_above = 0
		cnt_below = 0
		slope = 0
		if (xdist!=0):
			slope = float(Y - yind)/(X - xind)
			for u in range(0,width):
				for v in range(0,height):
					if ((v - slope*u + slope*X - Y)>0):
						den_above = den_above + ves_mask[u][v]
						cnt_above = cnt_above + 1
					elif ((v - slope*u + slope*X - Y)<0):
						den_below = den_below + ves_mask[u][v]
						cnt_below = cnt_below + 1
					else:
					 	density = 1
					
			dabove = float(den_above)/cnt_above
			dbelow = float(den_below)/cnt_below
			density = float(dabove)/dbelow

		else:
			density = 0

		fv = [xdist, ydist, density]
		k = [0]
		featvec.append(fv)
		klabels.append(k)


# Last step : convert featvec and klabels to numpy arrays.

featvec = np.array(featvec)
klabels = np.array(klabels)
featvec = np.float32(featvec)
print featvec
print klabels

# SVM  Training---------------------------
SZ=20
#bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
trainData = featvec
responses = klabels
svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

# This code finds the feature vectors and trains an SVM for the detection of OD.

import numpy as np
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import os

mypath = './extended'
images = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]

featvector = []
labels = []

for w in images:
	od_mask = cv2.imread('./od/' + w)    # Reads od mask
	od_mask = cv2.cvtColor(od_mask, cv2.COLOR_BGR2GRAY)
	mac_mask = cv2.imread('./mac/' + w) # Reads macula mask
	mac_mask = cv2.cvtColor(mac_mask, cv2.COLOR_BGR2GRAY)
	height, width = od_mask.shape
	height = int(height/2)
	width = int(width/2)
	od_mask = cv2.resize(od_mask, (height, width))
	mac_mask = cv2.resize(mac_mask, (height, width))
	m = np.where(od_mask>0) # Marks the OD region
	orig = cv2.imread('./extended/' + w) # Reads original image
	
	orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	orig=cv2.resize(orig,(height, width))
	ves_mask = cv2.imread('./vessels/' + w) # Reads vessel mask
	ves_mask = cv2.cvtColor(ves_mask, cv2.COLOR_BGR2GRAY)
	ves_mask=cv2.resize(ves_mask,(height, width))

# Find the horizontal line of symmetry : mrow contains the row index that has minimum difference in density on either side-----------------
	mdif = 10000
	mrow = 0
	for r in range(int(2*height/5), int(3*height/5)):
		vps_ab = len(np.where(ves_mask[0:r] == 255)[0])
		nop_ab = r*width
		vps_bel = len(np.where(ves_mask[r:height] == 255)[0])
		nop_bel = height*width - nop_ab
		den_ab = float(vps_ab)/nop_ab
		den_bel = float(vps_bel)/nop_bel
		dens = abs(den_ab - den_bel)
		if (dens< mdif):
			mdif = dens
			mrow = r
			       
	print 'row of equal density:',mrow
	cv2.line(ves_mask, (0,mrow), (width, mrow), 255)

#---------------------------------------------------------------------------------------------
# positive samples
	rands = np.random.choice(np.shape(m[0])[0], 30, replace=False)
	print 'random indices chosen:', rands
	for i in rands:
		origcopy = orig
		ves_maskcopy = ves_mask
		mask = np.zeros((width, height), np.uint8)
	  	cv2.circle(mask,(m[1][i], m[0][i]), 25, 255, -1)
		print 'maximum value in mask:', np.max(mask)
		masked_data = cv2.bitwise_and(origcopy,origcopy , mask=mask)
		int_sum = np.sum(masked_data) # 1st feature
			
		origcopy1 = orig
		mask1 = np.zeros((width,height), np.uint8)
		cv2.circle(mask1, (m[1][i], m[0][i]), 45, 255, -1)
		masked1 = cv2.bitwise_and(origcopy1, origcopy1, mask = mask1)
		int_sum1 = np.sum(masked1)   # 2nd feature

		ves_masked = cv2.bitwise_and(ves_maskcopy, ves_maskcopy, mask = mask)
		no_of_ves_pixels = len(np.where(ves_masked == 255)[0])
		no_of_total_pixels = len(np.where(mask == 255)[0])
		density = float(no_of_ves_pixels)/no_of_total_pixels   # 3nd feature
#-----------------------------------------------------------------------------------------------------------------------------------------
		dis = abs(mrow - m[0][i])    # 4th feature

#----------------------------------------------------------------------------------------------------------------
		fv = [int_sum, int_sum1, density, dis]  # Feature vector
		k = [1]     # Positive samples
		print fv
		featvector.append(fv)
		labels.append(k)
#----------------------------------------------------------------------------------------------------
# Negative samples
	print 'Negative samples'
	origcopy2 = orig
	outp = cv2.bitwise_and(origcopy2, origcopy2, mask = od_mask)
	negs = cv2.bitwise_and(outp, outp,mask = mac_mask)
	n = np.where(negs == 0) # These are indices of points that are non- macula, non- od
	rrnds = np.random.choice(np.shape(n[0])[0], 30, replace = False) # Randomly pick 30 negative samples
	for j in rrnds:
		origcopy = orig
		ves_maskcopy = ves_mask
		mask = np.zeros(( width, height), np.uint8)
	  	cv2.circle(mask,(n[1][j], n[0][j]), 25, 255, -1)
		print 'maximum value in mask:', np.max(mask)
		masked_data = cv2.bitwise_and(origcopy,origcopy , mask=mask)
		int_sum = np.sum(masked_data) # 1st feature
		
		origcopy1 = orig
		mask1 = np.zeros(( width, height), np.uint8)
		cv2.circle(mask1, (n[1][j], n[0][j]), 45, 255, -1)
		masked1 = cv2.bitwise_and(origcopy1, origcopy1, mask = mask1)
		int_sum1 = np.sum(masked1)   # 2nd feature

		ves_masked = cv2.bitwise_and(ves_maskcopy, ves_maskcopy, mask = mask)
		no_of_ves_pixels = len(np.where(ves_masked == 255)[0])
		no_of_total_pixels = len(np.where(mask == 255)[0])
		density = float(no_of_ves_pixels)/no_of_total_pixels   # 3nd feature
		
#-----------------------------------------------------------------------------------------------------------------------------------------
		dis = abs(mrow - n[0][j])    # 4th feature

#----------------------------------------------------------------------------------------------------------------
		fv = [int_sum, int_sum1, density, dis]  # Feature vector
		k = [0]     # Negative samples
		featvector.append(fv)
		labels.append(k)
		print 'fv:', fv
# Last step : convert featvec and klabels to numpy arrays.

featvector = np.array(featvector)
labels = np.array(labels)
featvector = np.float32(featvector)
print featvector
print labels

# SVM  Training---------------------------
SZ=20
#bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
trainData = featvector
responses = labels
svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_OD_data.dat')

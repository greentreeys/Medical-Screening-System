# OD Testing

#-----------------------------------------------------------------------------------

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from os import listdir
from os.path import isfile, join
import os


mypath = './extended'
images = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]

mypath1 = './vessels' 
vessmaps = [ g for g in sorted(listdir(mypath1)) if isfile(join(mypath,g))]


for w in images:
	print w
	orig = cv2.imread('./extended/' + w)
	orig = orig[:,:,2]
	height, width = orig.shape[:2]   
	height=(int)(height/2)
	width=(int)(width/2)
	orig=cv2.resize(orig,(height, width))
	finalorig = orig
	
	od_cands = [] # This array will contain all the od candidates
# Find the horizontal line of symmetry : mrow contains the row index that has minimum difference in density on either side-----------------
	ves_mask = cv2.imread('./vessels/' + w)
	ves_mask = cv2.cvtColor(ves_mask, cv2.COLOR_BGR2GRAY)
	ves_mask = cv2.resize(ves_mask, (height, width))
	mdif = 100
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

# Processing on the image--------------------------------------------------------------------------------------------------------------
	resized = orig
	scale=1
	delta=0
	ddepth=-1
	blurred = resized
	blurred=cv2.GaussianBlur(resized,(25,25),1)

# Derivatives in both directions not too helpful 
# -------------------------------------------------------------------------------

	horizontaledge = cv2.Scharr(blurred,ddepth,1, 0,7)
	ret, horizontaledge  = cv2.threshold(horizontaledge,150,255,cv2.THRESH_TOZERO)
	verticaledge = cv2.Scharr(blurred,ddepth,0, 1,7)
	ret1, verticaledge  = cv2.threshold(verticaledge,150,255,cv2.THRESH_TOZERO)

#--------------------------------------------------------------------------------

	nz = np.nonzero(resized)
	feat1 = np.zeros(horizontaledge.shape[:2])
	feat1[nz[0][:],nz[1][:]] = (abs(verticaledge[nz[0][:],nz[1][:]])-abs(horizontaledge[nz[0][:],nz[1][:]]))/(resized[nz[0][:],nz[1][:]])
	neg=np.where(feat1<0)
	feat1[neg[0][:],neg[1][:]]=0	

	feat2 = np.zeros(horizontaledge.shape[:2])
	feat2[nz[0][:],nz[1][:]] = (abs(horizontaledge[nz[0][:],nz[1][:]]) + abs(verticaledge[nz[0][:],nz[1][:]]))*(resized[nz[0][:],nz[1][:]])

#-------------------------------------------------------------------------------------------------------------------------

# Projection on the horizontal axis

	p=np.zeros(width)
	for i in range(12,width-13):
		submat = feat1[:,i-20:i+20]
		p[i] = submat.sum()
		 
# The function gives the indices of local maxima, these will be the od candidates' x coordinates
	maxarrx = argrelextrema(p, np.greater)
	
	xpts = maxarrx[0]

#------------------------------------------------------------------------------------
	
	yp = np.zeros((height,1))
	for p in xpts:
		if p > 15 and p < width - 16:
			for j in range(0,height):
				if (j>30) and (j<height-31):
					sub = feat2[j-30:j+30,p - 15:p + 15]
					yp[j] = np.sum(sub)
				elif (j<=30):
					sub = feat2[0:j,p - 15:p + 15]
					yp[j] = np.sum(sub)
				elif ( j>=height - 31):
					sub = feat2[j:height-1,p-15:p+15]
					yp[j] = np.sum(sub)

			ypos = np.argmax(yp)

		elif (p<=15): 
			for j in range(0,height):
				if (j>30) and (j<height-31):
					sub = feat2[j-30:j+30,0:p]
					yp[j] = np.sum(sub)
				elif (j<=30):
					sub = feat2[0:j,0:p]
					yp[j] = np.sum(sub)
				elif ( j>=height - 31):
					sub = feat2[j:height-1, 0:p]
					yp[j] = np.sum(sub)

			ypos = np.argmax(yp)
		
		elif (p>= width - 16):
			for j in range(0,height):
				if (j>30) and (j<height-31):
					sub = feat2[j-30:j+30,p:width-1]
					yp[j] = np.sum(sub)
				elif (j<=30):
					sub = feat2[0:j,p:width-1]
					yp[j] = np.sum(sub)
				elif ( j>=height - 31):
					sub = feat2[j:height-1, p:width-1]
					yp[j] = np.sum(sub)

			ypos = np.argmax(yp)
		od_can = [ p, ypos]
		od_cands.append(od_can)
		cv2.circle(orig, (p,ypos), 5 , 0, -1)
	od_cands = np.array(od_cands)
	
	
	fv = []
	results = np.zeros((od_cands.shape[0],1))
# Now find the feature vectors for each OD candidate, and test it on the SVM trained earlier.
	for c in range(0, od_cands.shape[0]):
		origcopy = orig
		mask = np.zeros((width, height), np.uint8)
                cv2.circle(mask,(od_cands[c][0], od_cands[c][1]), 25, 255, -1)
   	        masked_data = cv2.bitwise_and(origcopy,origcopy , mask=mask)
                int_sum = np.sum(masked_data) # 1st feature

		origcopy1 = orig
	        mask1 = np.zeros((width,height), np.uint8)
	        cv2.circle(mask1, (od_cands[c][0], od_cands[c][1]), 45, 255, -1)
	        masked1 = cv2.bitwise_and(origcopy1, origcopy1, mask = mask1)
	        int_sum1 = np.sum(masked1)   # 2nd feature
		
		ves_maskcopy = ves_mask
		ves_masked = cv2.bitwise_and(ves_maskcopy, ves_maskcopy, mask = mask)
	        no_of_ves_pixels = len(np.where(ves_masked == 255)[0])
	        no_of_total_pixels = len(np.where(mask == 255)[0])
	        density = float(no_of_ves_pixels)/no_of_total_pixels   # 3nd feature

		dis = abs(mrow - od_cands[c][1])    # 4th feature

		fvec = [int_sum, int_sum1, density, dis]  # Feature vector
		fv.append(fvec)
		
	fv = np.asarray(fv)
	fv = np.float32(fv)
	
	svm = cv2.SVM()
	svm.load('svm_OD_data.dat')
	results = svm.predict_all(fv)
	
	for i in range(0,od_cands.shape[0]):
		if (results[i] == [1]) and (i==4):
			cv2.circle(finalorig, (od_cands[i][0], od_cands[i][1]), 5, 255, -1)
			print od_cands[i][0]
			print od_cands[i][1]
	cv2.destroyAllWindows()	

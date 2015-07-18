# Macula Detection - Testing
#-----------------------------------------------------------------------------------

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from os import listdir
from os.path import isfile, join
import os

fv = []
mypath = './extended'
images = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]
mypath1 = './vessels' 
vessmaps = [ g for g in sorted(listdir(mypath1)) if isfile(join(mypath,g))]

f = open('res.txt', 'r')
d = f.readlines()
for g in d:
	print g
print d
print 'size of d: ', len(d)
inlistind = 0
for w in images:
	print w
	xi = inlistind+1
	Xod = int(d[xi])
	yi = xi+1
	Yod = int(d[yi])
	inlistind = yi+1
	print Xod
	print Yod
	orig = cv2.imread('./extended/' + w) # Reads original image
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	height, width = orig.shape[:2]
	height=(int)(height/2)
	width=(int)(width/2)
	orig = cv2.resize(orig,(height, width))
	
	if ( Xod > int(width/2)):
		sub = orig[Yod - 125: Yod + 125, Xod - 250:Xod]
	else:
	 	sub = orig[Yod - 125: Yod + 125, Xod :Xod + 250]

	u=0
	v=0
	while (u < width):
		while( v < height):
			xdist = abs(Xod - u)
			ydist = abs(Yod - v)
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
			fv  = [xdist, ydist,density]

			v = v+2
		u = u+2
	featvec.append(fv)

featvec = np.array(featvec)
svm = cv2.SVM()
svm.load('svm_data.dat')
results = svm.predict_all(fv)

import numpy as np;
import sys;
import os;
import random;
import math;

def invarience(image_array):
		p=np.amax(image_array,axis=0); #max pixel values across the rotated images
		return p;
		

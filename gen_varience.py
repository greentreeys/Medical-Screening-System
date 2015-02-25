#function [int_map, var_map] = gen_representation(img,coal_func,gmp_params,save_root,save_gmp,save_name)
import itertools;
import random;
import math;
import copy;
import numpy as np;
import cv2
def frange(x, y, jump):
	a=[]
	while x < y:
		a.append(x);
		x += jump
	return a;

################################################################################################################################33
def find_gmp(img,angles,coal_func,pv):	
	rows,cols = img.shape
	data={};
	i=0;j=len(angles);
	netimg=img;
	while(i<j):
		M = cv2.getRotationMatrix2D((int(pv[0]),int(pv[1])),angles[i],1);
		dst = cv2.warpAffine(img,M,(cols,rows))
		data[i]=dst;
		i+=1;
	kk=0
	mm=0
	while(kk<rows):
		mm=0;
		while(mm<cols):
			i=0;
			rr=[];
			j=len(angles);
			while(i<j):
				rr.append(data[i][kk][mm])
				i+=1;
			np.uint8(rr);
			if(coal_func=="max"):
				netimg[kk][mm]=max(rr);
			if(coal_func=="min"):
				netimg[kk][mm]=min(rr);
			if(coal_func=="sum"):
				l=sum(rr);
				if(l>255):
					l=255;
				netimg[kk][mm]=l;
			
			mm+=1;
		kk+=1;

	return netimg;
####################################################################################################################################33

def online_variance(new_data,curr_var,curr_iter,curr_mean):
	if curr_iter==1:
		new_mean = new_data;
		new_var = cv2.multiply(new_data,0);
		return new_mean,new_var;
	else:

		pa=cv2.subtract(new_data,curr_mean);
		pa=cv2.divide(pa,curr_iter,1);
		new_mean=cv2.add(pa,curr_mean);
		#new_mean = curr_mean + (new_data - curr_mean)/curr_iter;
		prev_S = cv2.multiply(curr_var,(curr_iter - 2));
	
		#
		pd1=cv2.subtract(new_data,curr_mean);
		pd2=cv2.subtract(new_data,new_mean);
		pd=cv2.multiply(pd1,pd2);
		new_S=cv2.add(pd,prev_S);
		#new_S = prev_S  + (new_data  - curr_mean) .* (new_data - new_mean);
		
		new_var=cv2.divide(new_S,curr_iter-1);
		
		#new_var = new_S/(curr_iter - 1);
		
		return (new_mean),(new_var);

def eqau(int_ma):
	int_map=np.uint64(int_ma);
	w=int(np.amax(int_map));
	int_map=cv2.multiply(int_map,255);
	int_map=cv2.divide(int_map,w);
	return np.uint8(int_map)

#####################################################################################################################################

def numel(a):
	return a.size;
def randperm(a,b):
	arr=range(1,a+1,1);
	return random.sample(arr,b);
def ind2sub(rows,cols,pv):
	R=[];
	C=[];
	for i in pv:
		r=math.floor((i-1)/rows);
		c=(i-1)%rows;
		R.append(int(r));
		C.append(int(c));
	return R,C;
def update_change(si,ei,sj,ej,source,dest):
	i=si;
	ii=0;
	while(i<ei):
		j=sj;
		jj=0;
		while(j<ej):
			source[i][j]=np.uint8(dest[ii][jj]);
			j+=1;
			jj+=1;
		i+=1;
		ii+=1;
class gmp_parameters(object):
	def __init__(self,min_angle,max_angle,step,pivots):
		self.pivots=pivots;
		self.min_angle=min_angle;
		self.max_angle=max_angle;
		self.step=step;
	
def gen_representation(img,coal_func,gmp_params,save_root,save_gmp,save_name):
	
	## later used variables;
	gmp_img=0;
	############################################################

	npv = gmp_params.pivots; 
	min_angle = gmp_params.min_angle; 
	step = gmp_params.step; 
	max_angle = gmp_params.max_angle; 
	angles = frange(min_angle,max_angle,step)
	###############################################################################
	rows,cols = img.shape;
	sz=[0,rows,cols];
	pv = randperm(numel(img),npv);
	R,C = ind2sub(rows,cols,pv);
	################################################################################
	curr_var = 0;
	curr_mean = 0;
	int_map = cv2.multiply(img,0);
	if save_gmp==1:
		gmp_img = np.zeros((2*(sz[1]+10), 2*(sz[2]+10),1),np.uint8);
	##################################################################################

	print rows,cols;
	
	for i in range(0,npv):
		print i
		new_data = find_gmp(img,angles,coal_func,[R[i],C[i]]);
		curr_mean,curr_var = online_variance(np.uint64(new_data),np.uint64(curr_var),i+1,np.uint64(curr_mean));
		int_map = cv2.add(np.uint64(int_map),np.uint64(new_data));
		var_map = curr_var;

			
	'''	w=int(np.amax(int_map));
		int_map=np.uint16(int_map);
		int_map=cv2.multiply(int_map,255);
		int_map=cv2.divide(int_map,w);
		int_map = np.uint8(int_map);
		
		cv2.imshow('int',int_map)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
		
		try:
			var_map = np.uint8(eqau(var_map));
			cv2.imshow('var',var_map)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		except:
			print var_map'''

		#print curr_var;
	############################################################################################
	int_map=eqau(int_map);
	var_map=eqau(var_map);

	cv2.imshow('int',int_map)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow('var',var_map)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=="__main__":
	img=np.uint8(cv2.imread('OrigImg0002.png',0));
	gmpp=gmp_parameters(-5,5,1,50);
	gen_representation(img,"max",gmpp,"a",1,"c");








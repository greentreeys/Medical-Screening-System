import cv2;
import numpy as np;
import sys;
import os;
import random;
import math;


#########################################################

class GMP_Params(object):

	def __init__(self,S_angle,E_angle,jump,func,pivots):

		self.angles=self.frange(S_angle,E_angle,jump);
		if(func=="max"):
			self.func=1;
		elif(func=="min"):
			self.func=-1;

		elif(func=="mean"):
			self.func=0;
		self.pivots=pivots;
	
	def frange(self,x,y,jump):
		arr=[];
		while x<y:
			arr.append(x);
			x+=jump;
		return arr;

class G_Params(object):

	def __init__(self,n_pivots,min_angle,step,max_angle):
		self.n_pivots = n_pivots;
		self.min_angle = min_angle; 
		self.step = step; 
		self.max_angle = max_angle; 
		self.angles = self.frange(min_angle,max_angle,step);
	
	def frange(self,x,y,jump):
		arr=[];
		while x<y:
			arr.append(x);
			x+=jump;
		return arr;

class SAL_Params(object):
	def __init__(self,numlevel,sal_level,cmin,cmax,cdelta,delmin,delmax,deltadel):
		self.numlevel = sal_params.numlevel;
		self.sal_level = sal_level;
		self.cmin = cmin;
		self.cmax = cmax;
		self.cdelta = cdelta;
		self.delmin = delmin;
		self.delmax = delmax;
		self.deltadel = deltadel;

##########################################################

class Nikhil_Implementation(object):

	def __init__(self,img_name):
		
		## reading image in gray scale
		self.img=cv2.imread(img_name);
		self.img=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		## converting it into Float of 32 bits 
		self.img=self.img.astype(float);



	## frange to generate the range of numbers
	def frange(self,x,y,jump):
		arr=[];
		while x<=y:
			arr.append(x);
			x+=jump;
		return arr;

	def mat2gray(self,image_version):
		image=image_version.copy();
		mi=np.amin(image);
		ma=np.amax(image);
		image=np.subtract(image,mi);
		image=np.divide(image,ma-mi);
		image=np.multiply(image,255);
		return np.uint8(image);


	def g_varience(self,gen_params):
		rows,cols=self.img.shape;
		numel=rows*cols;
		pivots=random.sample(range(0,numel-1),gen_params.n_pivots)
		pivots=[np.unravel_index(i, self.img.shape) for i in pivots]
		curr_var = 0;
		curr_mean = 0;
		int_map = np.zeros(shape=(rows,cols));
		reference_gparams=GMP_Params(gen_params.min_angle,gen_params.max_angle,gen_params.step,"max",[0,0]);
		for i in range(0,gen_params.n_pivots):
			print i;
			reference_gparams.pivots=[pivots[i][0],pivots[i][1]];
			new_data = self.generate_GMP(reference_gparams);
			curr_mean,curr_var = self.o_varience(new_data,curr_var,i+1,curr_mean); 
			int_map = np.add(int_map,new_data);
		
		var_map = curr_var;
		int_map = self.mat2gray(int_map);
		var_map = self.mat2gray(var_map);
		return int_map,var_map;




	def o_varience(self,new_data,curr_var,curr_iter,curr_mean):
		if(curr_iter==1):
			return new_data,0;
		else:
			new_mean = np.add(curr_mean,np.divide(np.subtract(new_data,curr_mean),curr_iter));
			prev_S = np.multiply(curr_var,np.subtract(curr_iter,2));
			new_S = np.add(prev_S,np.multiply(np.subtract(new_data,curr_mean),np.subtract(new_data,new_mean)));
			new_var = np.divide(new_S,(curr_iter - 1));
			return new_mean,new_var;


	def generate_GMP(self,gmp_params):
		rows,cols=self.img.shape;
		len_angles=len(gmp_params.angles);
		data=np.zeros(shape=(len_angles,rows,cols))
		for i in range(0,len_angles):
			M = cv2.getRotationMatrix2D((int(gmp_params.pivots[1]),int(gmp_params.pivots[0])),gmp_params.angles[i],1);
			dst = np.asarray(cv2.warpAffine(self.img,M,(cols,rows)));
			data[i,:,:]=np.asarray(dst);
		if(gmp_params.func==1):
			p=np.amax(data,axis=0);
			return p;
		if(gmp_params.func==-1):
			return np.amin(dst,axis=0);
		if(gmp_params.func==0):
			return np.sum(dst,axis=0);
	
	def pyramid(self,image_orig,sal_params,mode,scale):
		image=image_orig.copy();
		rows,cols=image.shape;
		image=cv2.resize(image,(int(rows*scale),int(cols*scale)));
		image=image.astype(float);
		#############################################################################
		numlevel = sal_params.numlevel;
		sal_level = sal_params.sal_level;
		cmin = sal_params.cmin;
		cmax = sal_params.cmax;
		cdelta = sal_params.cdelta;
		delmin = sal_params.delmin;
		delmax = sal_params.delmax;
		deltadel = sal_params.deltadel;
		##############################################################################
		data={};
		sal_sz =None;
		sz=[rows,cols];
		for i in range(1,sal_level+1):
			if i==1:
				sal_sz = sz;
			else:
				sal_sz = [math.ceil(sal_sz(1)/2),math.ceil(sal_sz(2)/2)];

		for i in range(1,num_level+1):
			if i ==1:
				data[i] = image;
			else:
				data[i] = cv2.pyrDown(data[i-1]);

		for i in range(1,num_level+1):
			data[i] = cv2.resize(data[i],sal_sz);
		irange=frange(cin,cmax,cdelta);
		newdata={};
		counter=0;
		for i in irange:
			jrange=frange(i+delmin,i+delmax,deltadel);
			for j in jrange:
				if i>=j:
					continue;
				if j>=numlevels:
					continue;
				temp = np.subtract(data[i],data[j]);
				if(mode==1):
					temp[temp<0]=0;
				else:
					temp[temp>0]=0;
				temp=np.absolute(temp)
				newdata[counter]=temp;
				counter+=1;
		diff_maps=np.zeros(shape=(counter,rows,cols));
		for i in range(0,counter):
			diff_maps[i,:,:]=newdata[i];
		ittimap=np.sum(diff_maps,axis=0);
		return ittimap;



		
###########################################################
if __name__ == "__main__":
	obj=Nikhil_Implementation('a.png');
	gen_params=G_Params(200,-5,1,5)
	int_map,var_map=obj.g_varience(gen_params);
	#gmp_params=GMP_Params(-5,5,1,"max",[576.0/2,720.0/2]);
	#image=obj.generate_GMP(gmp_params);
	#print image.shape;
	cv2.imshow('int_map',int_map.copy())
	cv2.imshow('var_map',var_map)
	cv2.waitKey(0)
	#cv2.destroyAllWindows()


import cv2;
import numpy as np;
import sys;
import os;
import random;
import math;

def Bcosfire_response(Blurshift_array, params,t):
    length,rows,cols=Blurshift_array.shape;

    rho_list=params[2,:];       #list of rhos from the filter
    
    max_rho= np.amax(rho_list); #max among the rho values

    sigmar=max_rho/3;           #sigma cap in equation 6 of the paper

    rs=np.ones((rows,cols));

    wi=np.zeros((length));
    
    wsum=0;
    
    for i in range (0,length):

        wi[i]=math.exp(-rho_list[i]*rho_list[i]/(2*sigmar*sigmar));#wi in equation 6 of the paper
        wsum=wsum+wi[i];
        
    print wi;
    
    wsum=1/wsum;                #the power 1/sigma wi in equation 6
    
    for x in range (0,rows):
        for y in range (0,cols):
            for i in range (0,length):
                
                p=math.pow(Blurshift_array[i,x,y],wi[i]);          #Si raised to wi in eqaution 6
                rs[x,y]=rs[x,y]*p;                                 #the multiplication in equation 6
            rs[x,y]=math.pow(rs[x,y],wsum);                        #rs in equation 6 with the product raised to wsum
            
    return rs;                                                     #return the final response as in equation 6

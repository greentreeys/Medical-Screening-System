import cv2;
import numpy as np;
import sys;
import os;
import random;
import math;

def Bcosfire_response(Blurshift_array, params,t):
    length,rows,cols=Blurshift_array.shape;
    print length;
    rho_list=params[2,:];
    print rho_list;
    max_rho= np.amax(rho_list);
    print max_rho;
    sigmar=max_rho/3;
    rs=np.ones((rows,cols));
    wi=np.zeros((length));
    wsum=0;
    for i in range (0,length):
        wi[i]=math.exp(-rho_list[i]*rho_list[i]/(2*sigmar*sigmar));
        wsum=wsum+wi[i];
    print wi;
    wsum=1/wsum;
    print wsum;
    for x in range (0,rows):
        for y in range (0,cols):
            for i in range (0,length):
                p=math.pow(Blurshift_array[i,x,y],wi[i]);
                rs[x,y]=rs[x,y]*p;
            rs[x,y]=math.pow(rs[x,y],wsum);
    return rs;

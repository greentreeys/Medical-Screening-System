import numpy as np
import cv2
from numpy import *


#### System Config function definition (incomplete)
def SystemConfig():
    params = np.zeros((1,1),dtype=('f4,(1,1)float64,(1,11)float64,(1,1)float64,(1,1)float64'))
    params.dtype.names = ('ht', 'COSFIRE', 'invariance','detection','inputfilter')
    params[0]['ht']=1
    print(params)
    
    


### BCOSFIRE fuction definition (incomplete)
def BCOSFIRE(im,SYMM,ASYMM,pre_th,th):
    ###protype pattern
    x=101
    y=101
    line=np.zeros((201,201), np.uint8)
    line[:,x]=255

    ####symmetric filter parameters
    print("hello1")
    preprocess(im,pre_th)
    
    
        
        
        
#### preprocessing stage
def preprocess(im,pre_th):
    B, G, R = cv2.split(im)
    #### preparing mask
    pre_th=0.4;
    lab_image = cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab_image);
    L=L/100;
    mask =(1-(L < pre_th))*255*255;

    #####
    im=G;
    bigimage,smallimage=getBigimg(G,mask)
    where_are_NaNs = isnan(bigimage)
    bigimage[where_are_NaNs] = 0
    
    
    cv2.imshow("w",bigimage1)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(bigimage)
    
    
    



def getBigimg(img,mask):
    sizex,sizey=img.shape;
    bigimg = np.zeros((sizex+100,sizey+100),np.float32)
    bigimg[50:(50+sizex), 50:(50+sizey)]=img
    
    bigmask = np.zeros((sizex+100,sizey+100),np.float32)
    bigmask[50:(50+sizex), 50:(50+sizey)]=mask
    

    ####creates artificial extension of image
    bigimg=fakepad(bigimg,bigmask,5,10)
    smallimg = bigimg[50:(50+sizex), 50:(50+sizey)];
    return(bigimg,smallimg)
    
    #smallimg = bigimg[50:(50+sizex), 50:(50+sizey)];

def fakepad(img,mask,erosionsize,iterations):
    nrows,ncols=mask.shape
    mask[0,:] =     np.zeros((1, ncols),np.float32);
    mask[nrows-1,:] = np.zeros((1, ncols),np.float32);
    mask=normalize(mask);

    ####missed something
    
    #############eroding image
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)); #structural element of size 0*0

    #mask = cv2.erode(mask,kernel,iterations = 1); #erosion of image

    ####### multipying image
    dilated = img* mask;

    oldmask=mask.astype(int);
    
    

    filter_1=np.ones((3,3),np.float32)

    filter_rows,filter_cols=np.nonzero(filter_1>0);
    filter_rows = filter_rows - 1;
    filter_cols = filter_cols - 1;


    for i in xrange(iterations):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); #structural element of size 0*0
        newmask = cv2.dilate(mask,kernel,iterations = 1); #dilate of image
        outerborder=(newmask.astype(int)) & ~oldmask;
        rows,cols=np.nonzero(outerborder>0);
        c=cols.shape;
        
        
        for j in xrange(len(cols)):
             col = cols[j]
             row = rows[j]
             filtered=[]
             for k in xrange(len(filter_cols)):
                  filtercol = filter_cols[k];
                  filterrow = filter_rows[k];
      
                  pixelrow = row + filterrow;
                  pixelcol = col + filtercol;
                  
                  if (pixelrow <= nrows & pixelrow>=0 & pixelcol<=ncols & pixelcol>=0 & oldmask[pixelrow,pixelcol]):
                      filtered=[dilated[pixelrow,pixelcol]]



            

             dilated[row,col]=np.mean(filtered, axis=0)
    oldmask=newmask
    return(dilated);

                      
                              

                      
            
        
        
                      
                      
        
    
        


    

    
    
    

    
    
def normalize(img):
    max_val=np.amax(img)
    r,c=img.shape
    for i in xrange(r):
        for j in xrange(c):
            if (img[i][j]==0):
                img[i][j]=0;
            else:
                img[i][j]=img[i][j]/max_val

    return(img)
                
    
    
    
    



    


    

    
    
    

    
    
    
   
    











###reading image
img = cv2.imread('F:/3-2/honors-2/DRIVE/test/images/02_test.tif',1);
img=img.astype('float32');
im=(img+1)/255.0;

######filter parameters
filter_params = np.zeros((2,),dtype=('f4,f4,f4,f4'))
filter_params.dtype.names = ('sigma', 'len', 'sigma0','alpha')
filter_params[:] = [(2.4,8,3,0.7),(1.8,22,2,0.1)]

##function
BCOSFIRE(im,filter_params[0],filter_params[1],0.5,37)
#SystemConfig()






#cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()



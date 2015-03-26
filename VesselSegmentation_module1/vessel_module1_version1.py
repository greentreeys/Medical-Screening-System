import numpy as np
import cv2
import math

def params():
    params={'ht':1,'COSFIRE':{'rholist':[0,2,4,6,8],'eta':[2.6180],'t1':[0],'t2':[0.4],'sigma0':[0.5],'alpha':[0.1167],\
                          'mintupleweight':[0.5],'ouputfunction':'geomentricmean','blurringfuction':'max',\
                          'weightingsigma':[6.7946],'t3':[0]},'invariance':{'rotation':{'psilist':[0,0.261799387799149,\
                                                                                               0.523598775598299,0.785398163397448,1.04719755119660,\
                                                                                               1.30899693899575,1.57079632679490,1.83259571459405,\
                                                                                               2.09439510239320,2.35619449019235,2.61799387799149,\
                                                                                               2.87979326579064]},'scale':{'upsilonlist':1},'reflection':0}\
        ,'detection':{'mindistance':8},'inputfilter':{'name':'DoG','DoG':{'polaritylist':[1],'sigmalist':[2.4],'sigmaratio':[0.5],'halfwaverct':[0]}},'symmetric':1}

    return(params)

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

def DoGBankResponse(img,params):
    sz1,sz2=img.shape;
    #print(len(params['sigmalist']))
    #params_sigmalist=1
    #params_polaritylist=1
    #print(params['sigmaratio'])
    #DoGBank=np.zeros((sz1,sz2,len(params['sigmalist']),len(params['polaritylist'])),dtype=np.uint8)
    #for i in xrange(len(params['sigmalist'])):
      #  for j in xrange(len(params['polaritylist'])):

    sigma=params['inputfilter']['DoG']['sigmalist'][0]
    sigma_ratio=params['inputfilter']['DoG']['sigmaratio'][0]
    onff=0
    #DoGBank=getDoG(img,params['DoG']['sigmalist'][0],params['polaritylist'][0],params['sigmaratio'][0],0,params['halfwaverct'][i])  ########change paramenters
    DoGBank=getDoG(img,sigma,1,sigma_ratio)       
    
    return(DoGBank)

def fspecial(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def getDoG(img,sigma,onoff,sigmaRatio):
    ##create diff of Gaussian kernal
    sz=round((sigma*3) * 2 + 1)
    
    #print(sz)
    g1 = fspecial((sz,sz),sigma)
    g2 = fspecial((sz,sz),sigma*sigmaRatio);

    if (onoff==1):
        G=g2-g1;  ### difference of guassians
    else:
        G=g1-g2;

    #print(np.amax(G))
    
       
    dst = cv2.filter2D(img*255,-1,G) ###convolution of image
    return(dst)

def configcosfire(img,params,x,y):
    #rho=2;
    
    angle=np.linspace(1,360,360)*(math.pi/180);
    #temp=[]
    r=1
    rho=2*r
    max1=np.amax(img)
    temp=np.zeros((360,1),dtype=np.uint8)
    x1=np.zeros((360,1),dtype=np.uint8)
    y1=np.zeros((360,1),dtype=np.uint8)
    sam=np.zeros((rho*4,1),dtype=np.uint8)
    phase=[]
    #print(rho)
    for th in range(360):

        x1[th]=x+abs(round(rho*math.cos(angle[th])))
        y1[th]=y+abs(round(rho*math.sin(angle[th])))

    
        #temp[th,0]=img[x1,y1]
        #if(temp[th,0]==max1):
         #   phase=angle[th]
          #  print(phase)
    
    x11=unique(x1,rho)
    x22=unique(y1,rho)
    
    #array['r'][0]=rho
    #print(array)
    philist=[]
    for i in xrange(rho*4+1):
        if(img[x11[i],x22[i]]==max1):
            xq=x11[i].astype(float)
            yq=x22[i].astype(float)
            philist=(math.atan(yq/xq))
            print(philist)

            

    




def unique(mat,rho):
    cnt=0
    sam=np.zeros((rho*4+1,1),dtype=np.uint8)
    for i in xrange(359):
        if(mat[i]!=mat[i+1]):
            sam[cnt,0]=mat[i,0]
            cnt=cnt+1

    #sam[(rho*4),0]=sam[0,0]
    #sam=np.insert(sam,[24,0],[sam[0,0]])  
    return(sam)
        

        
        
            
            
    
    
    

image = cv2.imread('F:/3-2/honors-2/DRIVE/test/images/02_test.tif',0)
syparams=params()
img1=DoGBankResponse(image,syparams)


x=100;
y=100;
line1=np.zeros((201,201), dtype=np.uint8);
line1[:,x]=255;
line2=normalize(line1);

template_r=DoGBankResponse(line2,syparams)

cosfire_points=configcosfire(template_r,syparams,x,y)

cv2.imwrite('F:/DogResponse.tif', img1) 




#cv2.namedWindow("w",0)
#cv2.imshow("w",img1);
cv2.waitKey(0)

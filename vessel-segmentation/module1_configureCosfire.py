import numpy as np
import cv2
import math
###symmetric case
sigma = 4
sigma0 = 0.5
flen=8

###asymmetric case
sigma_a = 3
sigma0_a = 0.5
flen_a=12

#############################functions

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

def DoGResponse(image,sigma,sigma0):
    ##create diff of Gaussian kernal
    sz=round((sigma*3)*2+1)
    g1 = fspecial((sz,sz),sigma)
    g2 = fspecial((sz,sz),sigma*sigma0)
    G=g2-g1
    DoGResp = cv2.filter2D(image*255,-1,G) ###convolution of image
    return(DoGResp)

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



def configurecosfire(image,x,y,flen,sigma,sigma0,filt):
    ##create tuple matrix
    #print image[1,1]
    if (filt=='sym'):
         tuples=np.zeros((4,2*flen-1));
    else:
        tuples=np.zeros((4,flen));
        
    
   
    angle=np.linspace(1,360,360)*(math.pi/180)
    th=np.amax(image)

    tuples[0,:]=sigma0;
    tuples[1,:]=sigma;
    
    cnt=1
    for i in xrange(flen):##flen
        rho=2*i
       
        count=0
        if(rho==0):
            theta=0;
            tuples[2,0]=0
            tuples[3,0]=0
        else:
            for theta in angle:
                x1=x+(rho*math.cos(theta))
                y1=y+(rho*math.sin(theta))

                x_1=math.modf(x1)
                y_1=math.modf(y1)

                if((x_1[0]==float(0)) & (y_1[0]==float(0))):
                    
                    if(image[x_1[1],y_1[1]]>th-10):
                        tuples[2,cnt]=rho
                        tuples[3,cnt]=theta
                        cnt=cnt+1;


                        
    return tuples;
                        
                        
                    
                       
       
    #print tuples.shape
    #print tuples
                
                            

        
    
    



####symmetric prototype
x,y=100,100
prototype_sym=np.zeros((201,201),dtype=np.uint8)
prototype_sym[:,x]=255
prototype_sym=normalize(prototype_sym);
template_sym=DoGResponse(prototype_sym,sigma,sigma0)

###asymmetric prototype

prototype_sym[100:201,100]=0
prototype_asym=prototype_sym
template_asym=DoGResponse(prototype_asym,sigma_a,sigma0_a)


###read image
#img=cv2.imread('1.jpg',0);
#img1=DoGResponse(img,sigma,sigma0)



###configiring cosfire

sym_struct=configurecosfire(template_sym,x,y,flen,sigma,sigma0,'sym')
print(sym_struct)
#print(sym_struct.shape)
asym_struct=configurecosfire(template_asym,x,y,flen_a,sigma_a,sigma0_a,'asym')
print(asym_struct)
#print(asym_struct.shape)


#cv2.namedWindow("w",0)
#cv2.imshow("w",im)
cv2.waitKey(0)

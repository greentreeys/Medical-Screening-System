from final_module1 import *
from bs_paper import *
from Bcosfire_response import *
from rotation_invarience import *

m,n=sym_struct.shape
print sym_struct
print asym_struct
m1,n1=asym_struct.shape
No_of_rotations=12
sigma0_sym=0.5
sigma0_asym=0.5
alpha_sym=0.01
alpha_asym=0.01

bsf=np.zeros((n,img1.shape[0],img1.shape[1]))
resp=np.zeros((No_of_rotations,img1.shape[0],img1.shape[1]))
bsf1=np.zeros((n1,img1.shape[0],img1.shape[1]))
resp1=np.zeros((No_of_rotations,img1.shape[0],img1.shape[1]))

for orien in range(No_of_rotations):
    cnt=0;
    for i in xrange(n):
        print sym_struct[0,i],sym_struct[2,i],sym_struct[3,i]
        bs = blurshift(img1,sigma0_sym,sym_struct[2,i],sym_struct[3,i]+math.pi*orien/No_of_rotations,alpha_sym)
        bsf[cnt,:,:]=bs
        cnt  = cnt + 1
        print cnt

    resp[orien,:,:] = Bcosfire_response(bsf,sym_struct,0);
    print '***'
    
    cnt1=0;
    for i1 in xrange(n1):
            
        bs1 = blurshift(img1,sigma0_asym,asym_struct[2,i1],asym_struct[3,i1]+math.pi*orien/No_of_rotations,alpha_asym)
        bsf1[cnt1,:,:]=bs1
        cnt1  = cnt1 + 1
        print cnt1
    
    resp1[orien,:,:] = Bcosfire_response(bsf1,asym_struct,1);


final = invarience(resp)
final1=invarience(resp1)
p=(final*255)+(final1*255)
q=p.astype(np.uint8)
ret,thresh1 = cv2.threshold(q,30,255,cv2.THRESH_BINARY)
cv2.imshow("w",thresh1)
cv2.waitKey(0)



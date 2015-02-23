import numpy as np
import cv2
def inpaint_vessels(img,mask,radius,method):
	img = cv2.imread(img)
	mask = cv2.imread(mask,0)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations=1)
        if method == 1:
		dst = cv2.inpaint(img,mask,radius,cv2.INPAINT_TELEA)
        else:
		dst = cv2.inpaint(img,mask,radius,cv2.INPAINT_NS)
        cv2.imshow('ori',img)
	cv2.imshow('dst',dst)
	cv2.imwrite('output_inpaint.jpg', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

inpaint_vessels('img.jpg','mask.jpg',11,1)

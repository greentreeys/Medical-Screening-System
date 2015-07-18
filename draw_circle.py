# Creates a template for the OD
import numpy as np
import cv2
from PIL import Image

#mgvess = cv2.imread('bp.jpg')
sz = 100
rad = 50
img = Image.new('RGBA', (100, 100), (255, 0, 0, 0))
pixels = img.load() # create the pixel map
 
for i in range(sz-1):    # for every pixel:
    for j in range(sz-1):
        pixels[i,j] = (0, 0, 0)


xc = (int)((sz+1)/2)
yc = (int)((sz+1)/2)
img = np.asarray(img)
cv2.circle(img, (xc,yc), rad, [255,255,255,0], 100)
cv2.imwrite('circle.jpg', img)
cv2.imshow('ll', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


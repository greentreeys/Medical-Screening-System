#!/usr/bin/python
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys

def inpaint_retina(source_path,mask_path,dest_path,img_ext):
        print "Hello"
        sources = [ f for f in listdir(source_path) if f.endswith(img_ext) ]
        print "Number of files to process is ",len(sources),"."
        masks = [ f for f in listdir(mask_path) if f.endswith(img_ext) ]
        kernel = np.ones((7,7),np.uint8)
        for (s,m) in zip(sources,masks):
                img = cv2.imread(join(source_path,s))
                mask = cv2.imread(join(mask_path,m),0)
                mask = cv2.threshold(mask,0, 255, cv2.THRESH_BINARY)[1]
                mask = cv2.dilate(mask,kernel,iterations = 1)
                dst = cv2.inpaint(img,mask,55,cv2.INPAINT_NS)
                cv2.imwrite(join(dest_path,s),dst)
                print "File ",s, " processed.\n"
        return

a = sys.argv[1]
b = sys.argv[2]
c = sys.argv[3]
d = sys.argv[4]

inpaint_retina(a, b, c, d)

#Example python inpaint_retina.py /home/ujjwal/images/ /home/ujjwal/masks  /home/ujjwal/results  .png

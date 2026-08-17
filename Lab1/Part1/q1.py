import cv2
import numpy as np 

img = cv2.imread('images/img1.gif') 
print(img)


img256 = cv2.resize(img,(256,256) , interpolation=cv2.INTER_AREA)
img128 = cv2.resize(img,(128,128) , interpolation=cv2.INTER_AREA)


cv2.imwrite("output/img1(256).png", img256)
cv2.imwrite("output/img1(128).png", img128)
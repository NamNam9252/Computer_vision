import cv2
import numpy as np 
from pathlib import Path

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "output"

img256 = cv2.imread(str(output_dir / "img1(256).png"))
img128 = cv2.imread(str(output_dir / "img1(128).png"))

# nearset 
# linear 
# cubic

imgs = [img128 , img256]

methods =[cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_CUBIC]
size = (512,512)

for y in range(0,2):
    for x in range(0,3):
        cv2.imwrite(str(output_dir / f"img{y}_{x}.png"),cv2.resize(imgs[y],size,interpolation=methods[x]))



def nearsetNeighbour( img : np.array  , sizeh : int  ,sizew : int  ):
    h,w,c = img.shape
    result  = np.zeros((sizeh , sizew ,c) ,dtype = np.uint8)

    for i in range(sizeh):
        for j in range(sizew):
            oldi = int(i*h/sizeh)
            oldj = int (j*w/sizew)
            result[i][j] = img[oldi][oldj]
    return result


cv2.imwrite(str(output_dir / "img_128_nearest.png") , nearsetNeighbour(img128,512,512))
cv2.imwrite(str(output_dir / "img_256_nearest.png") , nearsetNeighbour(img256,512,512))
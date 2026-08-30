import numpy as np

def nearestNeighbour( img : np.array  , sizeh : int  ,sizew : int  ):
    h,w,c = img.shape
    result  = np.zeros((sizeh , sizew ,c) ,dtype = np.uint8)
    for i in range(sizeh):
        for j in range(sizew):
            oldi = int(i*h/sizeh)
            oldj = int (j*w/sizew)
            result[i][j] = img[oldi][oldj]
    return result


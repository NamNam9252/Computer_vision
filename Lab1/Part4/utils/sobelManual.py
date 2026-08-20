import numpy as np 
import cv2
from . import convolution2D


def sobelManual(img :np.ndarray):
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    sobel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    
    # Convolution
    gx = convolution2D.convolution2D(img.astype(np.float32), sobel_x)
    gy = convolution2D.convolution2D(img.astype(np.float32), sobel_y)
    
    return gx, gy
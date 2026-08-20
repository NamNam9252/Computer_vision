import numpy as np 
import cv2


def convolution2D(image :np.ndarray, kernel :np.ndarray):
    #Get Dimensions of the image and kernel 
    img_h , img_w  = image.shape
    kernel_h , kernel_w = kernel.shape

    #Calculate the padding size
    pad_h = kernel_h //2
    pad_w = kernel_w //2

    #Pad the image with zeros using loop 
    padded_image = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    for i in range(img_h):
        for j in range(img_w):
            padded_image[i + pad_h, j + pad_w] = image[i, j]

    #Create an empty output image
    output_image = np.zeros((img_h, img_w))
    
    #Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output_image[i, j] = np.sum(region * kernel)

    return output_image

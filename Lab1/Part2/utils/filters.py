import numpy as np
from utils import convolution2D 

def gaussianKernel(size: int , sigma: float)->np.ndarray:
    if size%2 ==0:
        raise ValueError("Size must be an odd integer.")
    #create a grid of (x,y) coordinates
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    #calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    #normalize the kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussianFilter(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    kernel = gaussianKernel(size, sigma)
    filtered_image = convolution2D.convolution2D(image, kernel)
    return filtered_image


def weightedAverageFilter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    filtered_image = convolution2D.convolution2D(image, kernel)
    return filtered_image


def medianKernel(size: int) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")
    kernel = np.ones((size, size), dtype=np.float32)
    return kernel

def medianFilter(image:np.ndarray , size:int)->np.ndarray:
    kernel = medianKernel(size)
    pad_size = size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+size, j:j+size]
            filtered_image[i, j] = np.median(region)

    return filtered_image


def boxKernel(size: int) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Size must be an odd integer.")
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return kernel

def boxFilter(image: np.ndarray, size: int) -> np.ndarray:
    kernel = boxKernel(size)
    filtered_image = convolution2D.convolution2D(image, kernel)
    return filtered_image

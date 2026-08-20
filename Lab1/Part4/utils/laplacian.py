import numpy as np
import cv2
from . import convolution2D

def laplacian(image: np.ndarray):
    """
    Perform edge detection using the Laplacian operator.

    Parameters:
    - image: Input grayscale image (numpy array).

    Returns:
    - laplacian: The second-order derivative of the image.
    """
    # Compute the Laplacian (second-order derivative) manually using a kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    laplacian = convolution2D.convolution2D(image, kernel)
    laplacian = np.abs(laplacian)

    return laplacian
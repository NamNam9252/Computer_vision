import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import convolution2D as conv2D
from utils import  sobelManual as sobel

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("Part2/images/img1.png", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("Part2/images/img2.png", cv2.IMREAD_GRAYSCALE), SIZE) 

gx1, gy1 = sobel.sobelManual(img1)
gx2, gy2 = sobel.sobelManual(img2)

plt.figure(figsize=(10,8))

plt.subplot(2,3,1)
plt.imshow(img1, cmap="gray")
plt.title("Image 1")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(np.abs(gx1), cmap="gray")
plt.title("X Derivative")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(np.abs(gy1), cmap="gray")
plt.title("Y Derivative")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(img2, cmap="gray")
plt.title("Image 2")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(np.abs(gx2), cmap="gray")
plt.title("X Derivative")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(np.abs(gy2), cmap="gray")
plt.title("Y Derivative")
plt.axis("off")

plt.tight_layout()
plt.show()



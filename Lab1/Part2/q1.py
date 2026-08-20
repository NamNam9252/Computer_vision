import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import filters


# Weighted-average kernel
kernel_weighted = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]], dtype=np.float32) / 16



imgns1 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img3.png" , cv2.IMREAD_GRAYSCALE)
imgns2 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img4.png" , cv2.IMREAD_GRAYSCALE)



# Image 1: Salt & Pepper noisy
box1 = filters.boxFilter(imgns1, 3)
weighted1 = filters.weightedAverageFilter(imgns1, kernel_weighted)
gaussian1 = filters.gaussianFilter(imgns1, 3, 1)
median1 = filters.medianFilter(imgns1, 3)


# Image 2: Gaussian noisy
box2 = filters.boxFilter(imgns2, 3)
weighted2 = filters.weightedAverageFilter(imgns2, kernel_weighted)
gaussian2 = filters.gaussianFilter(imgns2, 3, 1)
median2 = filters.medianFilter(imgns2, 3)


plt.figure(figsize=(10,8))

plt.subplot(3,2,1)
plt.imshow(imgns1, cmap="gray")
plt.title("Noisy Image 1")
plt.axis("off")

plt.subplot(3,2,2)
plt.imshow(box1, cmap="gray")
plt.title("Box Filter")
plt.axis("off")

plt.subplot(3,2,3)
plt.imshow(weighted1, cmap="gray")
plt.title("Weighted Average")
plt.axis("off")

plt.subplot(3,2,4)
plt.imshow(gaussian1, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(3,2,5)
plt.imshow(median1, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))

plt.subplot(3,2,1)
plt.imshow(imgns2, cmap="gray")
plt.title("Noisy Image 2")
plt.axis("off")

plt.subplot(3,2,2)
plt.imshow(box2, cmap="gray")
plt.title("Box Filter")
plt.axis("off")

plt.subplot(3,2,3)
plt.imshow(weighted2, cmap="gray")
plt.title("Weighted Average")
plt.axis("off")

plt.subplot(3,2,4)
plt.imshow(gaussian2, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(3,2,5)
plt.imshow(median2, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
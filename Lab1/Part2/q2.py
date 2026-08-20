import cv2
import numpy as np
import matplotlib.pyplot as plt

sizes = [3, 5, 7]

imgns1 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img3.png" , cv2.IMREAD_GRAYSCALE)
imgns2 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img4.png" , cv2.IMREAD_GRAYSCALE)

# Weighted-average kernels
k3 = np.ones((3,3), np.float32) / 9
k5 = np.ones((5,5), np.float32) / 25
k7 = np.ones((7,7), np.float32) / 49

kernels = [k3, k5, k7]

# -------- Image 1: Salt & Pepper --------
box1 = [cv2.blur(imgns1, (k,k)) for k in sizes]
weighted1 = [cv2.filter2D(imgns1, -1, k) for k in kernels]
gaussian1 = [cv2.GaussianBlur(imgns1, (k,k), 0) for k in sizes]
median1 = [cv2.medianBlur(imgns1, k) for k in sizes]

# -------- Image 2: Gaussian Noise --------
box2 = [cv2.blur(imgns2, (k,k)) for k in sizes]
weighted2 = [cv2.filter2D(imgns2, -1, k) for k in kernels]
gaussian2 = [cv2.GaussianBlur(imgns2, (k,k), 0) for k in sizes]
median2 = [cv2.medianBlur(imgns2, k) for k in sizes]


plt.figure(figsize=(12,10))

for i in range(3):
    plt.subplot(4,3,i+1)
    plt.imshow(box1[i], cmap="gray")
    plt.title(f"Box {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+4)
    plt.imshow(weighted1[i], cmap="gray")
    plt.title(f"Weighted {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+7)
    plt.imshow(gaussian1[i], cmap="gray")
    plt.title(f"Gaussian {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+10)
    plt.imshow(median1[i], cmap="gray")
    plt.title(f"Median {sizes[i]}x{sizes[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))

for i in range(3):
    plt.subplot(4,3,i+1)
    plt.imshow(box2[i], cmap="gray")
    plt.title(f"Box {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+4)
    plt.imshow(weighted2[i], cmap="gray")
    plt.title(f"Weighted {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+7)
    plt.imshow(gaussian2[i], cmap="gray")
    plt.title(f"Gaussian {sizes[i]}x{sizes[i]}")
    plt.axis("off")

for i in range(3):
    plt.subplot(4,3,i+10)
    plt.imshow(median2[i], cmap="gray")
    plt.title(f"Median {sizes[i]}x{sizes[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
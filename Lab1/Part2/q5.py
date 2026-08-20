import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


sizes = [3, 5, 7]

img1 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img3.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img4.png", cv2.IMREAD_GRAYSCALE)

imgns1 = img1 

img3=img1
img4=img2

imgns2 = img2 
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


# 5. Absolute-difference images

def difference(original, filtered):
    return cv2.absdiff(original, filtered)

# Image 1 - Salt & Pepper
diff_box1 = difference(img3, box1[0])
diff_weighted1 = difference(img3, weighted1[0])
diff_gaussian1 = difference(img3, gaussian1[0])
diff_median1 = difference(img3, median1[0])

# Image 2 - Gaussian
diff_box2 = difference(img4, box2[0])
diff_weighted2 = difference(img4, weighted2[0])
diff_gaussian2 = difference(img4, gaussian2[0])
diff_median2 = difference(img4, median2[0])


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(diff_box1, cmap="gray")
plt.title("Box 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(diff_weighted1, cmap="gray")
plt.title("Weighted 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(diff_gaussian1, cmap="gray")
plt.title("Gaussian 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(diff_median1, cmap="gray")
plt.title("Median 3x3 Difference")
plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(diff_box2, cmap="gray")
plt.title("Box 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(diff_weighted2, cmap="gray")
plt.title("Weighted 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(diff_gaussian2, cmap="gray")
plt.title("Gaussian 3x3 Difference")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(diff_median2, cmap="gray")
plt.title("Median 3x3 Difference")
plt.axis("off")

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import cv2
import numpy as np

imgo = cv2.imread(r"S:\Computer Vision\Lab1\Part1\images\img1.gif")
img128 = cv2.imread(r"S:\Computer Vision\Lab1\Part1\output\img_128_nearest.png")
img256 = cv2.imread(r"S:\Computer Vision\Lab1\Part1\output\img_256_nearest.png")


def edges(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    return np.sqrt(gx**2 + gy**2)


edge_original = edges(imgo)
edge_128 = edges(img128)
edge_256 = edges(img256)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(edge_original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edge_128, cmap="gray")
plt.title("128 × 128 Nearest")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge_256, cmap="gray")
plt.title("256 × 256 Nearest")
plt.axis("off")

plt.tight_layout()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("Part2/images/img1.png", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("Part2/images/img2.png", cv2.IMREAD_GRAYSCALE), SIZE) 
# Image 1
gx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
gy1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)

# Image 2
gx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
gy2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude and direction

# Image 1
magnitude1 = np.sqrt(gx1**2 + gy1**2)
direction1 = np.arctan2(gy1, gx1)

# Image 2
magnitude2 = np.sqrt(gx2**2 + gy2**2)
direction2 = np.arctan2(gy2, gx2)

# Convert direction from radians to degrees
direction1 = np.degrees(direction1)
direction2 = np.degrees(direction2)


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(magnitude1, cmap="gray")
plt.title("Image 1 - Gradient Magnitude")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(direction1, cmap="gray")
plt.title("Image 1 - Gradient Direction")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(magnitude2, cmap="gray")
plt.title("Image 2 - Gradient Magnitude")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(direction2, cmap="gray")
plt.title("Image 2 - Gradient Direction")
plt.axis("off")

plt.tight_layout()
plt.show()
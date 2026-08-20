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

# 3. Binary edge maps using different thresholds

thresholds = [50, 100, 150]

# Image 1
edge1_50 = (magnitude1 > 50) * 255
edge1_100 = (magnitude1 > 100) * 255
edge1_150 = (magnitude1 > 150) * 255

# Image 2
edge2_50 = (magnitude2 > 50) * 255
edge2_100 = (magnitude2 > 100) * 255
edge2_150 = (magnitude2 > 150) * 255


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(edge1_50, cmap="gray")
plt.title("Threshold = 50")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edge1_100, cmap="gray")
plt.title("Threshold = 100")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edge1_150, cmap="gray")
plt.title("Threshold = 150")
plt.axis("off")

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(edge2_50, cmap="gray")
plt.title("Threshold = 50")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edge2_100, cmap="gray")
plt.title("Threshold = 100")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edge2_150, cmap="gray")
plt.title("Threshold = 150")
plt.axis("off")

plt.tight_layout()
plt.show()

print('''A low threshold detects both strong and weak intensity changes, resulting in more edges but also unwanted edges and noise. As the threshold increases, unwanted edges are removed and the edge map becomes cleaner. However, a very high threshold can cause weak but meaningful edges to disappear. Therefore, an intermediate threshold generally provides a good balance between detecting useful edges and avoiding unwanted edges.''')
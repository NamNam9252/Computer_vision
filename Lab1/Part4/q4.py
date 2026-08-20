import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import laplacian

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("Part2/images/img1.png", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("Part2/images/img2.png", cv2.IMREAD_GRAYSCALE), SIZE) 


# 4. Second-order derivative using Laplacian

lap1 = laplacian.laplacian(img1)
lap2 = laplacian.laplacian(img2)

# Binary edge maps
edge1 = (lap1 > 30).astype(np.uint8) * 255
edge2 = (lap2 > 30).astype(np.uint8) * 255

# Display
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(lap1, cmap='gray')
plt.title("Second-Order Derivative")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(edge1, cmap='gray')
plt.title("Second-Order Edge Map")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img2, cmap='gray')
plt.title("Image 2")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(lap2, cmap='gray')
plt.title("Second-Order Derivative")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(edge2, cmap='gray')
plt.title("Second-Order Edge Map")
plt.axis("off")

plt.tight_layout()
plt.show()


print('''The first-order derivative detects edges by measuring the rate of intensity change, whereas the second-order derivative detects rapid changes in the gradient. The Laplacian produces thin and sharp edge responses and can highlight fine details more strongly. However, it is more sensitive to noise and may produce unwanted or double-edge responses. The first-order derivative generally produces more stable and easily interpretable edges, while the second-order derivative provides sharper edge localization but is more sensitive to small intensity variations.''')
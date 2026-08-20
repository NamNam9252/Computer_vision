import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("Part2/images/img1.png", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("Part2/images/img2.png", cv2.IMREAD_GRAYSCALE), SIZE) 


# 4. Second-order derivative using Laplacian

lap1 = cv2.Laplacian(img1, cv2.CV_64F)
lap2 = cv2.Laplacian(img2, cv2.CV_64F)

# Convert to positive magnitude
lap1 = np.abs(lap1)
lap2 = np.abs(lap2)

# Binary edge maps
edge1 = (lap1 > 30).astype(np.uint8) * 255
edge2 = (lap2 > 30).astype(np.uint8) * 255


# 5. LoG and Canny Edge Detection

def log_edges(img):
    # Gaussian smoothing
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    
    # Laplacian
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Convert to edge map using threshold
    lap = np.abs(lap)
    edges = (lap > 20).astype(np.uint8) * 255
    
    return edges


# LoG
log1 = log_edges(img1)
log2 = log_edges(img2)

# Canny
canny1 = cv2.Canny(img1, 100, 200)
canny2 = cv2.Canny(img2, 100, 200)


# Display results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(log1, cmap='gray')
plt.title("LoG Edge Map")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(canny1, cmap='gray')
plt.title("Canny Edge Map")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img2, cmap='gray')
plt.title("Image 2")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(log2, cmap='gray')
plt.title("LoG Edge Map")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(canny2, cmap='gray')
plt.title("Canny Edge Map")
plt.axis("off")

plt.tight_layout()
plt.show()
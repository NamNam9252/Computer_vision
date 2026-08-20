import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread(
    r"S:\Computer Vision\Lab1\Part2\images\img1.png",
    cv2.IMREAD_GRAYSCALE
)

snp = cv2.imread(
    r"S:\Computer Vision\Lab1\Part2\images\saltandpeper.png",
    cv2.IMREAD_GRAYSCALE
)

gau = cv2.imread(
    r"S:\Computer Vision\Lab1\Part2\images\gaussian.png",
    cv2.IMREAD_GRAYSCALE
)

# Create an empty result image
result = np.zeros(img1.shape, dtype=np.float32)

# Left half → Salt & Pepper
for x in range(0, 256):
    for y in range(0, 512):
        result[x][y] = img1[x][y] + 0.4 * snp[x][y]

# Right half → Gaussian
for x in range(256, 512):
    for y in range(0, 512):
        result[x][y] = img1[x][y] + 0.4 * gau[x][y]

# Keep pixel values between 0 and 255
result = np.clip(result, 0, 255).astype(np.uint8)

plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()



# Apply all four filters

# 1. Box filter
box = cv2.blur(result, (3,3))

# 2. Weighted-average filter
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]], dtype=np.float32) / 16
weighted = cv2.filter2D(result, -1, kernel)

# 3. Gaussian filter
gaussian = cv2.GaussianBlur(result, (3,3), 0)

# 4. Median filter
median = cv2.medianBlur(result, 3)


# Display all results
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(result, cmap="gray")
plt.title("Noisy Complete Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(box, cmap="gray")
plt.title("Box Filter")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(weighted, cmap="gray")
plt.title("Weighted Average")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(median, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()



print('''The Median filter gives the best overall visual result for the complete image, as it removes the salt-and-pepper noise effectively while preserving the butterfly's edges and fine details. However, the Gaussian filter performs better for the Gaussian-noise region. Therefore, the Median filter provides the best overall compromise when a single filter must be applied to the complete image.''')




###Ques7 

# Region-wise smoothing

result_region = np.zeros_like(result)

# Left half - Salt & Pepper noise -> Median filter
result_region[0:256, :] = cv2.medianBlur(result[0:256, :], 3)

# Right half - Gaussian noise -> Gaussian filter
result_region[256:512, :] = cv2.GaussianBlur(result[256:512, :], (3,3), 0)


# Display result
plt.figure(figsize=(6,6))
plt.imshow(result_region, cmap="gray")
plt.title("Region-wise Smoothing")
plt.axis("off")
plt.show()
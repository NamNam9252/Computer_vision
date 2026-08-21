import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images
img1 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img5.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img6.png", cv2.IMREAD_GRAYSCALE)

# Make both images the same size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert to float
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)

# -----------------------------
# LOW PASS
# -----------------------------

low = cv2.GaussianBlur(
    img1,
    (21, 21),
    2
)

# -----------------------------
# HIGH PASS
# -----------------------------

blur2 = cv2.GaussianBlur(
    img2,
    (21, 21),
    5
)

high = img2 - blur2

# -----------------------------
# HYBRID
# -----------------------------

hybrid = low + high 

# Keep values between 0 and 255
hybrid = np.clip(hybrid, 0, 255)

# Convert back to uint8
low = np.uint8(np.clip(low, 0, 255))
high_display = np.uint8(np.clip(high + 128, 0, 255))
hybrid = np.uint8(hybrid)

# -----------------------------
# DISPLAY
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap="gray")
plt.title("Image 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(low, cmap="gray")
plt.title("Low Pass")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img2, cmap="gray")
plt.title("Image 2")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(blur2, cmap="gray")
plt.title("Blurred Image 2")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(high_display, cmap="gray")
plt.title("High Pass")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(hybrid, cmap="gray")
plt.title("Hybrid Image")
plt.axis("off")

plt.tight_layout()
plt.show()
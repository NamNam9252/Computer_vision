import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Ky = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)

def magnitude(img):
    f = img.astype(np.float64)
    Ix = convolve2d(f, Kx, mode="same", boundary="symm")
    Iy = convolve2d(f, Ky, mode="same", boundary="symm")
    return np.sqrt(Ix**2 + Iy**2)

THRESHOLDS = [30, 80, 150]

fig, axes = plt.subplots(2, len(THRESHOLDS) + 1, figsize=(16, 8))
fig.suptitle("Binary Edge Maps at 3 Threshold Values", fontsize=13, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1 (boundaries)"), (img2, "Img2 (fine detail)")]):
    mag = magnitude(img)
    axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].set_title(label); axes[row, 0].axis("off")
    for col, t in enumerate(THRESHOLDS):
        binary = (mag > t).astype(np.uint8) * 255
        axes[row, col + 1].imshow(binary, cmap="gray")
        axes[row, col + 1].set_title(f"Threshold={t}")
        axes[row, col + 1].axis("off")

plt.tight_layout()
plt.savefig("output/q3_thresholds.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q3_thresholds.png")
plt.show()

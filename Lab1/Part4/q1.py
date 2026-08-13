import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)  # clear boundaries
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)  # fine details

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Ky = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)

def sobel_manual(img):
    """First-order derivatives via manual Sobel convolution (no cv2.Sobel)."""
    f = img.astype(np.float64)
    Ix = convolve2d(f, Kx, mode="same", boundary="symm")
    Iy = convolve2d(f, Ky, mode="same", boundary="symm")
    return Ix, Iy

cv2.imwrite("output/img1.png", img1)
cv2.imwrite("output/img2.png", img2)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("First-Order Derivatives (Manual Sobel)", fontsize=13, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1 (boundaries)"), (img2, "Img2 (fine detail)")]):
    Ix, Iy = sobel_manual(img)
    axes[row, 0].imshow(img, cmap="gray");                   axes[row, 0].set_title(f"{label}"); axes[row, 0].axis("off")
    axes[row, 1].imshow(np.clip(Ix + 128, 0, 255), cmap="gray"); axes[row, 1].set_title("Ix (x-derivative)"); axes[row, 1].axis("off")
    axes[row, 2].imshow(np.clip(Iy + 128, 0, 255), cmap="gray"); axes[row, 2].set_title("Iy (y-derivative)"); axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("output/q1_derivatives.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q1_derivatives.png")
plt.show()

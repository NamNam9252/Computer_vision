import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

Kx  = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Ky  = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)
Klap = np.array([[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]], dtype=np.float64)

def first_order(img):
    f = img.astype(np.float64)
    Ix = convolve2d(f, Kx, mode="same", boundary="symm")
    Iy = convolve2d(f, Ky, mode="same", boundary="symm")
    return np.sqrt(Ix**2 + Iy**2)

def second_order(img):
    """Manual Laplacian (no cv2.Laplacian)."""
    return convolve2d(img.astype(np.float64), Klap, mode="same", boundary="symm")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("First-order vs Second-order Derivatives", fontsize=13, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1"), (img2, "Img2")]):
    grad = first_order(img)
    lap  = second_order(img)
    lap_vis = np.clip(lap + 128, 0, 255).astype(np.uint8)

    axes[row, 0].imshow(img, cmap="gray");      axes[row, 0].set_title(f"{label}");              axes[row, 0].axis("off")
    axes[row, 1].imshow(grad, cmap="hot");      axes[row, 1].set_title("Gradient Magnitude");    axes[row, 1].axis("off")
    axes[row, 2].imshow(lap_vis, cmap="gray");  axes[row, 2].set_title("Laplacian (2nd order)"); axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("output/q4_second_order.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q4_second_order.png")
plt.show()

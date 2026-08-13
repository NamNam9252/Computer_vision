import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

Kx   = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Ky   = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)
Klap = np.array([[ 0, 1, 0], [ 1,-4, 1], [ 0, 1, 0]], dtype=np.float64)

def sobel_mag(img):
    f = img.astype(np.float64)
    return np.sqrt(convolve2d(f, Kx, mode="same", boundary="symm")**2 +
                   convolve2d(f, Ky, mode="same", boundary="symm")**2)

def sobel_binary(img, t=80):
    return (sobel_mag(img) > t).astype(np.uint8) * 255

def laplacian_map(img):
    return np.clip(convolve2d(img.astype(np.float64), Klap, mode="same", boundary="symm") + 128, 0, 255).astype(np.uint8)

def log_edges(img, sigma=2, thresh=10):
    k = int(6 * sigma + 1) | 1
    lap = convolve2d(cv2.GaussianBlur(img, (k, k), sigma).astype(np.float64),
                     Klap, mode="same", boundary="symm")
    return (np.abs(lap) > thresh).astype(np.uint8) * 255

methods = ["Original", "Sobel (binary)", "Laplacian", "LoG", "Canny"]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle("Edge Method Comparison — Localization, Continuity, Thickness, Fine Detail",
             fontsize=12, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1 (boundaries)"), (img2, "Img2 (fine detail)")]):
    maps = [
        img,
        sobel_binary(img),
        laplacian_map(img),
        log_edges(img),
        cv2.Canny(img, 50, 150),
    ]
    for col, (im, t) in enumerate(zip(maps, methods)):
        axes[row, col].imshow(im, cmap="gray")
        axes[row, col].set_title(f"{label}\n{t}" if col == 0 else t)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("output/q6_comparison.png", dpi=120, bbox_inches="tight")
print("Saved -> output/q6_comparison.png")
plt.show()

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

def sobel_binary(img, t=80):
    f = img.astype(np.float64)
    mag = np.sqrt(convolve2d(f, Kx, mode="same", boundary="symm")**2 +
                  convolve2d(f, Ky, mode="same", boundary="symm")**2)
    return (mag > t).astype(np.uint8) * 255

def log_edges(img, sigma=2, thresh=10):
    k = int(6 * sigma + 1) | 1
    lap = convolve2d(cv2.GaussianBlur(img, (k, k), sigma).astype(np.float64),
                     Klap, mode="same", boundary="symm")
    return (np.abs(lap) > thresh).astype(np.uint8) * 255

print("=" * 60)
print("Best edge detection method per image:")
print("=" * 60)
print("Img1 (clear boundaries) -> Canny")
print("  Reason: Strong edges, non-max suppression gives 1-pixel-thin,")
print("  well-localized, continuous boundaries. Hysteresis avoids noise.")
print()
print("Img2 (fine detail)      -> LoG")
print("  Reason: LoG detects zero-crossings at fine scales (small sigma)")
print("  preserving subtle texture edges lost by Canny's smoothing.")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Best Method Per Image", fontsize=13, fontweight="bold")

for row, (img, label, best_fn, best_label) in enumerate([
    (img1, "Img1 (boundaries)", lambda i: cv2.Canny(i, 50, 150),   "Canny (best)"),
    (img2, "Img2 (fine detail)", lambda i: log_edges(i, sigma=1.5), "LoG s=1.5 (best)"),
]):
    sobel = sobel_binary(img)
    best  = best_fn(img)
    axes[row, 0].imshow(img, cmap="gray");   axes[row, 0].set_title(label);       axes[row, 0].axis("off")
    axes[row, 1].imshow(sobel, cmap="gray"); axes[row, 1].set_title("Sobel");      axes[row, 1].axis("off")
    axes[row, 2].imshow(best, cmap="gray");  axes[row, 2].set_title(best_label);   axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("output/q7_best_method.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q7_best_method.png")
plt.show()

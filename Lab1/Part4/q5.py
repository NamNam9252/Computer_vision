import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

Klap = np.array([[0, 1, 0], [1,-4, 1], [0, 1, 0]], dtype=np.float64)

def log_edges(img, sigma=2, thresh=10):
    """Laplacian of Gaussian: blur then manual Laplacian, zero-crossings as edges."""
    k = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(img, (k, k), sigma).astype(np.float64)
    lap = convolve2d(blurred, Klap, mode="same", boundary="symm")
    return (np.abs(lap) > thresh).astype(np.uint8) * 255

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("LoG vs Canny Edge Detection", fontsize=13, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1 (boundaries)"), (img2, "Img2 (fine detail)")]):
    log  = log_edges(img, sigma=2, thresh=10)
    canny = cv2.Canny(img, 50, 150)

    axes[row, 0].imshow(img, cmap="gray");   axes[row, 0].set_title(f"{label}");       axes[row, 0].axis("off")
    axes[row, 1].imshow(log, cmap="gray");   axes[row, 1].set_title("LoG (s=2, t=10)"); axes[row, 1].axis("off")
    axes[row, 2].imshow(canny, cmap="gray"); axes[row, 2].set_title("Canny (50,150)");  axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("output/q5_log_canny.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q5_log_canny.png")
plt.show()

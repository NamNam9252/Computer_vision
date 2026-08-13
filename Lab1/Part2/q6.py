import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    mask  = np.random.random(img.shape)
    noisy[mask < prob / 2]     = 0
    noisy[mask > 1 - prob / 2] = 255
    return noisy

def add_gaussian_noise(img, sigma=25):
    return np.clip(img.astype(np.float64) + np.random.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def weighted_avg_kernel(k):
    h = np.arange(1, k // 2 + 2)
    w1d = np.concatenate([h, h[-2::-1]])
    k2d = np.outer(w1d, w1d).astype(np.float64)
    return k2d / k2d.sum()

def box_manual(img, k):
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return np.clip(convolve2d(img.astype(np.float64), kernel, mode="same", boundary="symm"), 0, 255).astype(np.uint8)

K = 5
img = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
mid = w // 2

mixed = np.hstack([add_salt_pepper(img[:, :mid]), add_gaussian_noise(img[:, mid:])])
cv2.imwrite("output/q6_mixed.png", mixed)

filters = {
    "Box":      lambda i: box_manual(i, K),
    "W-Avg":    lambda i: cv2.filter2D(i, -1, weighted_avg_kernel(K)),
    "Gaussian": lambda i: cv2.GaussianBlur(i, (K, K), 0),
    "Median":   lambda i: cv2.medianBlur(i, K),
}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Mixed-Noise Image with All 4 Filters", fontsize=13, fontweight="bold")
axes[0].imshow(mixed, cmap="gray"); axes[0].set_title("Mixed Noisy"); axes[0].axis("off")

for ax, (name, fn) in zip(axes[1:], filters.items()):
    result = fn(mixed)
    cv2.imwrite(f"output/q6_{name.lower()}.png", result)
    ax.imshow(result, cmap="gray"); ax.set_title(name); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q6_mixed_filters.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q6_mixed_filters.png")
plt.show()

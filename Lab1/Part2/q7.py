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

K = 5
img = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)
mid = img.shape[1] // 2

mixed = np.hstack([add_salt_pepper(img[:, :mid]), add_gaussian_noise(img[:, mid:])])

# Best filter for S&P noise: Median | Best for Gaussian noise: Gaussian
left  = cv2.medianBlur(mixed[:, :mid], K)       # Median on left half (S&P)
right = cv2.GaussianBlur(mixed[:, mid:], (K, K), 0)  # Gaussian on right half

region_wise = np.hstack([left, right])
cv2.imwrite("output/q7_regionwise.png", region_wise)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Region-wise Smoothing", fontsize=13, fontweight="bold")
for ax, (im, title) in zip(axes, [(mixed, "Mixed Noisy"), (region_wise, "Region-wise Filter"),
                                   (np.abs(img.astype(np.float64) - region_wise.astype(np.float64)), "Residual Error")]):
    ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q7_regionwise.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q7_regionwise.png")
plt.show()

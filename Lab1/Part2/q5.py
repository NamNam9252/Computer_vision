import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def weighted_avg_kernel(k):
    h = np.arange(1, k // 2 + 2)
    w1d = np.concatenate([h, h[-2::-1]])
    k2d = np.outer(w1d, w1d).astype(np.float64)
    return k2d / k2d.sum()

def box_manual(img, k):
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return np.clip(convolve2d(img.astype(np.float64), kernel, mode="same", boundary="symm"), 0, 255).astype(np.uint8)

K = 5
noisy1 = cv2.imread("output/noisy1_sp.png",    cv2.IMREAD_GRAYSCALE)
noisy2 = cv2.imread("output/noisy2_gauss.png", cv2.IMREAD_GRAYSCALE)

filters = {
    "Box":      lambda img: box_manual(img, K),
    "W-Avg":    lambda img: cv2.filter2D(img, -1, weighted_avg_kernel(K)),
    "Gaussian": lambda img: cv2.GaussianBlur(img, (K, K), 0),
    "Median":   lambda img: cv2.medianBlur(img, K),
}

for noisy, tag, title in [(noisy1, "sp", "Salt-Pepper"), (noisy2, "gauss", "Gaussian Noise")]:
    fig, axes = plt.subplots(1, len(filters), figsize=(16, 4))
    fig.suptitle(f"Absolute Difference D(x,y) = |Noisy - Filtered| — {title}", fontsize=12, fontweight="bold")

    for ax, (name, fn) in zip(axes, filters.items()):
        diff = np.abs(noisy.astype(np.float64) - fn(noisy).astype(np.float64))
        im = ax.imshow(diff, cmap="hot", vmin=0, vmax=255)
        ax.set_title(f"{name}\nmean={diff.mean():.2f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"output/q5_diff_{tag}.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> output/q5_diff_{tag}.png")

plt.show()

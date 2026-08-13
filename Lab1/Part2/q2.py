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
    out = convolve2d(img.astype(np.float64), kernel, mode="same", boundary="symm")
    return np.clip(out, 0, 255).astype(np.uint8)

KERNELS = [3, 7, 15]

noisy1 = cv2.imread("output/noisy1_sp.png",    cv2.IMREAD_GRAYSCALE)
noisy2 = cv2.imread("output/noisy2_gauss.png", cv2.IMREAD_GRAYSCALE)

filter_fns = {
    "Box":      lambda img, k: box_manual(img, k),
    "W-Avg":    lambda img, k: cv2.filter2D(img, -1, weighted_avg_kernel(k)),
    "Gaussian": lambda img, k: cv2.GaussianBlur(img, (k, k), 0),
    "Median":   lambda img, k: cv2.medianBlur(img, k),
}

for noisy, tag, title in [(noisy1, "sp", "Salt-Pepper"), (noisy2, "gauss", "Gaussian")]:
    fig, axes = plt.subplots(len(filter_fns), len(KERNELS) + 1, figsize=(14, 14))
    fig.suptitle(f"Filter Effect vs Kernel Size — {title} Noise", fontsize=13, fontweight="bold")

    for row, (fname, fn) in enumerate(filter_fns.items()):
        axes[row, 0].imshow(noisy, cmap="gray"); axes[row, 0].set_title("Noisy"); axes[row, 0].axis("off")
        for col, k in enumerate(KERNELS):
            result = fn(noisy, k)
            axes[row, col + 1].imshow(result, cmap="gray")
            axes[row, col + 1].set_title(f"{fname} k={k}")
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"output/q2_kernels_{tag}.png", dpi=120, bbox_inches="tight")
    print(f"Saved -> output/q2_kernels_{tag}.png")

plt.show()

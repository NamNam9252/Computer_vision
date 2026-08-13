import cv2
import numpy as np
from scipy.signal import convolve2d

def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 20 * np.log10(255.0 / np.sqrt(m))

def weighted_avg_kernel(k):
    h = np.arange(1, k // 2 + 2)
    w1d = np.concatenate([h, h[-2::-1]])
    k2d = np.outer(w1d, w1d).astype(np.float64)
    return k2d / k2d.sum()

def box_manual(img, k):
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return np.clip(convolve2d(img.astype(np.float64), kernel, mode="same", boundary="symm"), 0, 255).astype(np.uint8)

K = 5
img1 = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE)

noisy1 = cv2.imread("output/noisy1_sp.png",    cv2.IMREAD_GRAYSCALE)
noisy2 = cv2.imread("output/noisy2_gauss.png", cv2.IMREAD_GRAYSCALE)

filters = {
    "Box":      lambda img: box_manual(img, K),
    "W-Avg":    lambda img: cv2.filter2D(img, -1, weighted_avg_kernel(K)),
    "Gaussian": lambda img: cv2.GaussianBlur(img, (K, K), 0),
    "Median":   lambda img: cv2.medianBlur(img, K),
}

print("=" * 60)
print(f"{'Noise':<10} {'Filter':<12} {'MSE':>10} {'PSNR (dB)':>12}")
print("=" * 60)

for orig, noisy, noise_tag in [(img1, noisy1, "S&P"), (img2, noisy2, "Gaussian")]:
    for name, fn in filters.items():
        filtered = fn(noisy)
        m, p = mse(orig, filtered), psnr(orig, filtered)
        print(f"{noise_tag:<10} {name:<12} {m:>10.4f} {p:>12.4f}")
    print("-" * 60)

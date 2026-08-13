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

KERNELS = [3, 5, 7, 15]
img1 = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE)

def add_sp(img): 
    noisy = img.copy(); mask = np.random.random(img.shape)
    noisy[mask < 0.025] = 0; noisy[mask > 0.975] = 255; return noisy

def add_gauss(img):
    return np.clip(img.astype(np.float64) + np.random.normal(0, 25, img.shape), 0, 255).astype(np.uint8)

noisy1, noisy2 = add_sp(img1), add_gauss(img2)

print("=" * 70)
print(f"{'Noise':<10} {'Filter':<12} {'k':>4}  {'MSE':>10} {'PSNR (dB)':>12}")
print("=" * 70)

for orig, noisy, noise_tag in [(img1, noisy1, "S&P"), (img2, noisy2, "Gaussian")]:
    for k in KERNELS:
        filters = {
            "Box":      box_manual(noisy, k),
            "W-Avg":    cv2.filter2D(noisy, -1, weighted_avg_kernel(k)),
            "Gaussian": cv2.GaussianBlur(noisy, (k | 1, k | 1), 0),
            "Median":   cv2.medianBlur(noisy, k | 1),
        }
        for name, filtered in filters.items():
            m, p = mse(orig, filtered), psnr(orig, filtered)
            print(f"{noise_tag:<10} {name:<12} {k:>4}  {m:>10.4f} {p:>12.4f}")
    print("-" * 70)

print("\nRecommendation:")
print("  Salt-and-Pepper noise  -> Median filter    (removes impulse noise, preserves edges)")
print("  Gaussian noise         -> Gaussian filter  (matches noise distribution, best PSNR)")

import cv2
import numpy as np

def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 20 * np.log10(255.0 / np.sqrt(m))

def nearest_manual(img, out_h, out_w):
    """Nearest-neighbour resize without cv2.resize (vectorised)."""
    h, w = img.shape
    ry = np.floor(np.arange(out_h) * h / out_h).astype(int)
    rx = np.floor(np.arange(out_w) * w / out_w).astype(int)
    return img[np.ix_(ry, rx)]

TARGET = (512, 512)
original_512 = cv2.resize(
    cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE),
    TARGET, interpolation=cv2.INTER_CUBIC
)

entries = [
    ("128->512", "Nearest",  "output/img0_0.png"),
    ("128->512", "Linear",   "output/img0_1.png"),
    ("128->512", "Cubic",    "output/img0_2.png"),
    ("256->512", "Nearest",  "output/img1_0.png"),
    ("256->512", "Linear",   "output/img1_1.png"),
    ("256->512", "Cubic",    "output/img1_2.png"),
]

print("=" * 62)
print(f"{'Source':<10} {'Method':<20} {'MSE':>10} {'PSNR (dB)':>12}")
print("=" * 62)

results = []
for src, method, path in entries:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m, p = mse(original_512, img), psnr(original_512, img)
    results.append((src, method, m, p))
    print(f"{src:<10} {method:<20} {m:>10.4f} {p:>12.4f}")

img128 = cv2.imread("output/img1(128).png", cv2.IMREAD_GRAYSCALE)
manual = nearest_manual(img128, 512, 512)
m_m, p_m = mse(original_512, manual), psnr(original_512, manual)
print(f"{'128->512':<10} {'Nearest (manual)':<20} {m_m:>10.4f} {p_m:>12.4f}")
print("=" * 62)

best = min(results, key=lambda r: r[2])
print(f"\nBest: {best[1]} from {best[0]}  |  MSE={best[2]:.4f}  PSNR={best[3]:.4f} dB")

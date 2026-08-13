import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)

SIGMA, K = 15, 31

# Spatial domain
t0 = time.perf_counter()
lpf_s = cv2.GaussianBlur(img1, (K, K), SIGMA)
hpf_s = img2 - cv2.GaussianBlur(img2, (K, K), SIGMA)
hybrid_s = np.clip(lpf_s + hpf_s, 0, 255).astype(np.uint8)
t_spatial = time.perf_counter() - t0

# Frequency domain
def gauss_mask(shape, sigma):
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    return np.exp(-((X - w//2)**2 + (Y - h//2)**2) / (2 * sigma**2))

t0 = time.perf_counter()
H, W = img1.shape
lm = gauss_mask((H, W), SIGMA)
F1 = np.fft.fftshift(np.fft.fft2(img1))
F2 = np.fft.fftshift(np.fft.fft2(img2))
lpf_f = np.real(np.fft.ifft2(np.fft.ifftshift(F1 * lm)))
hpf_f = np.real(np.fft.ifft2(np.fft.ifftshift(F2 * (1 - lm))))
hybrid_f = np.clip(lpf_f + hpf_f, 0, 255).astype(np.uint8)
t_freq = time.perf_counter() - t0

print(f"Spatial domain time : {t_spatial*1000:.2f} ms")
print(f"Frequency domain time: {t_freq*1000:.2f} ms")

diff = np.abs(hybrid_s.astype(np.float64) - hybrid_f.astype(np.float64))
print(f"Mean pixel difference between methods: {diff.mean():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Spatial ({t_spatial*1000:.1f}ms) vs Frequency ({t_freq*1000:.1f}ms)", fontsize=12, fontweight="bold")
axes[0].imshow(hybrid_s, cmap="gray"); axes[0].set_title("Spatial Hybrid"); axes[0].axis("off")
axes[1].imshow(hybrid_f, cmap="gray"); axes[1].set_title("Frequency Hybrid"); axes[1].axis("off")
axes[2].imshow(diff, cmap="hot");     axes[2].set_title(f"Difference (mean={diff.mean():.2f})"); axes[2].axis("off")

plt.tight_layout()
plt.savefig("output/q5_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q5_comparison.png")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)

SIGMA = 15

def gaussian_mask(shape, sigma):
    h, w = shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

H, W = img1.shape
lpf_mask = gaussian_mask((H, W), SIGMA)
hpf_mask = 1 - lpf_mask

F1 = np.fft.fftshift(np.fft.fft2(img1))
F2 = np.fft.fftshift(np.fft.fft2(img2))

lpf1 = np.real(np.fft.ifft2(np.fft.ifftshift(F1 * lpf_mask)))
hpf2 = np.real(np.fft.ifft2(np.fft.ifftshift(F2 * hpf_mask)))

hybrid = np.clip(lpf1 + hpf2, 0, 255).astype(np.uint8)
hpf2_vis = np.clip(hpf2 + 128, 0, 255).astype(np.uint8)

cv2.imwrite("output/q4_hybrid_freq.png", hybrid)

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(f"Hybrid Image (frequency domain)  sigma={SIGMA}", fontsize=12, fontweight="bold")
titles = ["I1", "I2", "LPF(I1)", "HPF(I2) norm", "Hybrid H"]
imgs   = [img1, img2, lpf1, hpf2_vis, hybrid]
for ax, im, t in zip(axes, imgs, titles):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q4_hybrid_freq.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q4_hybrid_freq.png")
plt.show()

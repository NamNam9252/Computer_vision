import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)

SIGMA, K = 15, 31

lpf = cv2.GaussianBlur(img1, (K, K), SIGMA)
hpf = img2 - cv2.GaussianBlur(img2, (K, K), SIGMA)
hybrid = np.clip(lpf + hpf, 0, 255).astype(np.uint8)
hpf_vis = np.clip(hpf + 128, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Hybrid Image Summary — All Results", fontsize=14, fontweight="bold")

row0 = [(img1, "I1 — LPF source"), (img2, "I2 — HPF source"),
        (lpf, "LPF(I1)"), (hpf_vis, "HPF(I2) normalised")]
row1 = [(hybrid, "Hybrid (close view)"),
        (cv2.resize(hybrid, (128, 128)), "Hybrid (128px — far)"),
        (cv2.imread("output/q6_pyramid_blend.png", cv2.IMREAD_GRAYSCALE), "Pyramid Blend"),
        (cv2.imread("output/q7_bilateral_blend.png", cv2.IMREAD_GRAYSCALE), "Bilateral Pyramid")]

for ax, (im, t) in zip(axes[0], row0):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
for ax, (im, t) in zip(axes[1], row1):
    if im is None: ax.axis("off"); continue
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q8_summary.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q8_summary.png")
plt.show()

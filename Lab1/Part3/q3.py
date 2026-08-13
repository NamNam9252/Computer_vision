import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)

def make_hybrid(I1, I2, sigma, alpha=1.0, beta=1.0):
    k = int(6 * sigma + 1) | 1
    lpf = cv2.GaussianBlur(I1, (k, k), sigma)
    hpf = I2 - cv2.GaussianBlur(I2, (k, k), sigma)
    return np.clip(alpha * lpf + beta * hpf, 0, 255).astype(np.uint8), hpf

configs = [
    (5,  0.8, 1.2, "Small sigma"),
    (15, 1.0, 1.0, "Medium sigma (best)"),
    (30, 1.2, 0.8, "Large sigma"),
]

fig, axes = plt.subplots(len(configs), 4, figsize=(18, 4 * len(configs)))
fig.suptitle("Parameter Experiments — sigma / alpha / beta", fontsize=13, fontweight="bold")

for row, (sigma, alpha, beta, label) in enumerate(configs):
    hybrid, hpf = make_hybrid(img1, img2, sigma, alpha, beta)
    hpf_vis = np.clip(hpf + 128, 0, 255).astype(np.uint8)
    lpf = cv2.GaussianBlur(img1, (int(6 * sigma + 1) | 1, int(6 * sigma + 1) | 1), sigma)

    for col, (im, t) in enumerate([(lpf, f"LPF s={sigma}"), (hpf_vis, "HPF (norm)"),
                                    (hybrid, f"Hybrid a={alpha} b={beta}"),
                                    (cv2.resize(hybrid, (128, 128)), "Downsampled")]):
        axes[row, col].imshow(im, cmap="gray")
        axes[row, col].set_title(f"{label}\n{t}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("output/q3_experiments.png", dpi=120, bbox_inches="tight")
print("Saved -> output/q3_experiments.png")
plt.show()

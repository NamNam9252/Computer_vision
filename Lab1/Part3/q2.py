import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE).astype(np.float64)

SIGMA, K = 15, 31  # selected parameters
ALPHA, BETA = 1.0, 1.0

lpf1 = cv2.GaussianBlur(img1, (K, K), SIGMA)
hpf2 = img2 - cv2.GaussianBlur(img2, (K, K), SIGMA)

hybrid = np.clip(ALPHA * lpf1 + BETA * hpf2, 0, 255).astype(np.uint8)
hpf2_vis = np.clip(hpf2 + 128, 0, 255).astype(np.uint8)  # normalised for display only

cv2.imwrite("output/q2_hybrid.png", hybrid)

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle(f"Hybrid Image (spatial domain)  sigma={SIGMA}, k={K}, a={ALPHA}, b={BETA}", fontsize=12, fontweight="bold")
for ax, (im, t) in zip(axes, [(img1, "I1 (LPF source)"), (img2, "I2 (HPF source)"),
                               (lpf1, "LPF(I1)"), (hpf2_vis, "HPF(I2) normalised"),
                               (hybrid, "Hybrid H")]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q2_hybrid.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q2_hybrid.png")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

TARGET = (512, 512)
original = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)
original = cv2.resize(original, TARGET, interpolation=cv2.INTER_CUBIC)

labels = [
    ("128->512 Nearest", "output/img0_0.png"),
    ("128->512 Linear",  "output/img0_1.png"),
    ("128->512 Cubic",   "output/img0_2.png"),
    ("256->512 Nearest", "output/img1_0.png"),
    ("256->512 Linear",  "output/img1_1.png"),
    ("256->512 Cubic",   "output/img1_2.png"),
]

def edge_map(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)

fig, axes = plt.subplots(7, 2, figsize=(10, 28))
fig.suptitle("Visual Quality & Edge Preservation Comparison", fontsize=14, fontweight="bold")

axes[0, 0].imshow(original, cmap="gray");      axes[0, 0].set_title("Original");       axes[0, 0].axis("off")
axes[0, 1].imshow(edge_map(original), cmap="hot"); axes[0, 1].set_title("Original Edges"); axes[0, 1].axis("off")

for i, (title, path) in enumerate(labels):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    axes[i+1, 0].imshow(img, cmap="gray");           axes[i+1, 0].set_title(title);           axes[i+1, 0].axis("off")
    axes[i+1, 1].imshow(edge_map(img), cmap="hot"); axes[i+1, 1].set_title(f"{title} Edges"); axes[i+1, 1].axis("off")

plt.tight_layout()
plt.savefig("output/q5_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q5_comparison.png")
plt.show()

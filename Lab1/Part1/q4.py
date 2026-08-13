import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original and resize to 512x512 (same as Q3)
original = cv2.imread("s:/Computer Vision/Lab1/Part1/images/img1.gif")
original = cv2.resize(original, (512, 512), interpolation=cv2.INTER_CUBIC)
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # grayscale for clear error maps

# The 6 reconstructed images from Q2
labels = [
    ("128->512 Nearest",  "output/img0_0.png"),
    ("128->512 Linear",   "output/img0_1.png"),
    ("128->512 Cubic",    "output/img0_2.png"),
    ("256->512 Nearest",  "output/img1_0.png"),
    ("256->512 Linear",   "output/img1_1.png"),
    ("256->512 Cubic",    "output/img1_2.png"),
]

fig, axes = plt.subplots(6, 2, figsize=(10, 24))
fig.suptitle("Pixel-wise Error Maps", fontsize=16, fontweight="bold")

for row, (title, path) in enumerate(labels):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orig_f = original.astype(np.float64)
    img_f  = img.astype(np.float64)

    abs_err = np.abs(orig_f - img_f)          # absolute error
    sq_err  = (orig_f - img_f) ** 2           # squared error

    # Absolute error map
    axes[row, 0].imshow(abs_err, cmap="hot")
    axes[row, 0].set_title(f"{title}\nAbsolute Error  (mean={abs_err.mean():.2f})")
    axes[row, 0].axis("off")
    plt.colorbar(axes[row, 0].images[0], ax=axes[row, 0], fraction=0.046)

    # Squared error map
    axes[row, 1].imshow(sq_err, cmap="hot")
    axes[row, 1].set_title(f"{title}\nSquared Error  (mean={sq_err.mean():.2f})")
    axes[row, 1].axis("off")
    plt.colorbar(axes[row, 1].images[0], ax=axes[row, 1], fraction=0.046)

plt.tight_layout()
plt.savefig("output/q4_error_maps.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q4_error_maps.png")
plt.show()

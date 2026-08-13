import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
Ky = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)

def gradient(img):
    f = img.astype(np.float64)
    Ix = convolve2d(f, Kx, mode="same", boundary="symm")
    Iy = convolve2d(f, Ky, mode="same", boundary="symm")
    mag = np.sqrt(Ix**2 + Iy**2)
    ang = np.degrees(np.arctan2(Iy, Ix))
    return Ix, Iy, mag, ang

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Gradient Magnitude & Direction", fontsize=13, fontweight="bold")

for row, (img, label) in enumerate([(img1, "Img1"), (img2, "Img2")]):
    Ix, Iy, mag, ang = gradient(img)
    for col, (im, t) in enumerate([
        (img, f"{label}"),
        (np.clip(Ix + 128, 0, 255), "Ix"),
        (mag, "Magnitude"),
        (ang,  "Direction (deg)"),
    ]):
        cmap = "hsv" if col == 3 else "gray"
        axes[row, col].imshow(im, cmap=cmap)
        axes[row, col].set_title(t)
        axes[row, col].axis("off")
        if col == 2:
            plt.colorbar(axes[row, col].images[0], ax=axes[row, col], fraction=0.046)

plt.tight_layout()
plt.savefig("output/q2_gradient.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q2_gradient.png")
plt.show()

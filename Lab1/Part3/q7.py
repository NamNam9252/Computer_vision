import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

LEVELS = 5

def bilateral_pyramid(img, levels, d=9, sc=75, ss=75):
    """Build Gaussian-like pyramid using bilateral filter instead of pyrDown."""
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        blurred = cv2.bilateralFilter(pyr[-1].astype(np.uint8), d, sc, ss).astype(np.float32)
        pyr.append(cv2.resize(blurred, (blurred.shape[1]//2, blurred.shape[0]//2)))
    return pyr

def laplacian_bilateral(img, levels):
    gpyr = bilateral_pyramid(img, levels)
    lpyr = []
    for i in range(levels - 1):
        up = cv2.resize(gpyr[i + 1], (gpyr[i].shape[1], gpyr[i].shape[0]))
        lpyr.append(gpyr[i] - up)
    lpyr.append(gpyr[-1])
    return lpyr

def gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr

def reconstruct(lpyr):
    img = lpyr[-1]
    for lap in reversed(lpyr[:-1]):
        img = cv2.resize(img, (lap.shape[1], lap.shape[0])) + lap
    return np.clip(img, 0, 255).astype(np.uint8)

def blend_bilateral(img_l, img_r, levels):
    h, w = img_l.shape
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, w // 2:] = 1.0

    lp1 = laplacian_bilateral(img_l, levels)
    lp2 = laplacian_bilateral(img_r, levels)
    gm  = gaussian_pyramid(mask, levels)

    blended_pyr = [l1 * (1 - g) + l2 * g for l1, l2, g in zip(lp1, lp2, gm)]
    return reconstruct(blended_pyr)

result_bilateral = blend_bilateral(img1, img2, LEVELS)
cv2.imwrite("output/q7_bilateral_blend.png", result_bilateral)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Laplacian Pyramid with Bilateral Filter", fontsize=13, fontweight="bold")
axes[0].imshow(img1, cmap="gray"); axes[0].set_title("Img1"); axes[0].axis("off")
axes[1].imshow(result_bilateral, cmap="gray"); axes[1].set_title("Bilateral Pyramid Blend"); axes[1].axis("off")

plt.tight_layout()
plt.savefig("output/q7_bilateral.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q7_bilateral.png")
plt.show()

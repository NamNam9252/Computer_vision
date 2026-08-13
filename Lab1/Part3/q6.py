import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)

LEVELS = 5

def gaussian_pyramid(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr

def laplacian_pyramid(img, levels):
    gpyr = gaussian_pyramid(img, levels)
    lpyr = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gpyr[i + 1], dstsize=(gpyr[i].shape[1], gpyr[i].shape[0]))
        lpyr.append(gpyr[i] - up)
    lpyr.append(gpyr[-1])
    return lpyr

def reconstruct(lpyr):
    img = lpyr[-1]
    for lap in reversed(lpyr[:-1]):
        img = cv2.pyrUp(img, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return np.clip(img, 0, 255).astype(np.uint8)

def blend(img_l, img_r, levels):
    """Blend left half of img_l with right half of img_r using Laplacian pyramids."""
    h, w = img_l.shape
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, w // 2:] = 1.0

    lp1  = laplacian_pyramid(img_l, levels)
    lp2  = laplacian_pyramid(img_r, levels)
    gm   = gaussian_pyramid(mask, levels)

    blended_pyr = [l1 * (1 - gm) + l2 * gm for l1, l2, gm in zip(lp1, lp2, gm)]
    return reconstruct(blended_pyr)

result = blend(img1, img2, LEVELS)
cv2.imwrite("output/q6_pyramid_blend.png", result)

# Show pyramids for img1
gpyr = gaussian_pyramid(img1, LEVELS)
lpyr = laplacian_pyramid(img1, LEVELS)

fig, axes = plt.subplots(2, LEVELS + 1, figsize=(18, 6))
fig.suptitle("Gaussian & Laplacian Pyramids + Blended Result", fontsize=13, fontweight="bold")
for i, g in enumerate(gpyr):
    axes[0, i].imshow(g, cmap="gray"); axes[0, i].set_title(f"G[{i}]"); axes[0, i].axis("off")
for i, l in enumerate(lpyr):
    vis = np.clip(l + 128, 0, 255).astype(np.uint8)
    axes[1, i].imshow(vis, cmap="gray"); axes[1, i].set_title(f"L[{i}]"); axes[1, i].axis("off")

axes[0, LEVELS].imshow(result, cmap="gray"); axes[0, LEVELS].set_title("Blended"); axes[0, LEVELS].axis("off")
axes[1, LEVELS].axis("off")

plt.tight_layout()
plt.savefig("output/q6_pyramids.png", dpi=120, bbox_inches="tight")
print("Saved -> output/q6_pyramids.png")
plt.show()

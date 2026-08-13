import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = (512, 512)
img1 = cv2.resize(cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img2 = cv2.resize(cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE), SIZE)
img3 = cv2.resize(cv2.imread("images/imge3.gif", cv2.IMREAD_GRAYSCALE), SIZE)

def align(ref, src):
    """Align src to ref using ECC with translation warp."""
    warp = np.eye(2, 3, dtype=np.float32)
    _, warp = cv2.findTransformECC(
        ref.astype(np.float32), src.astype(np.float32),
        warp, cv2.MOTION_TRANSLATION,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    )
    return cv2.warpAffine(src, warp, SIZE, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

img2_aligned = align(img1, img2)
img3_aligned = align(img1, img3)

cv2.imwrite("output/img1.png",          img1)
cv2.imwrite("output/img2_aligned.png",  img2_aligned)
cv2.imwrite("output/img3_aligned.png",  img3_aligned)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Pair Alignment using ECC", fontsize=13, fontweight="bold")
for ax, (im, t) in zip(axes[0], [(img1, "Img1 (ref)"), (img2, "Img2 (before)"), (img2_aligned, "Img2 (aligned)")]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
for ax, (im, t) in zip(axes[1], [(img1, "Img1 (ref)"), (img3, "Img3 (before)"), (img3_aligned, "Img3 (aligned)")]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")

plt.tight_layout()
plt.savefig("output/q1_alignment.png", dpi=150, bbox_inches="tight")
print("Saved -> output/q1_alignment.png")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# READ IMAGES
# =========================================================

img1 = cv2.imread(
    r"S:\Computer Vision\Lab1\Part3\images\img5.png",
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    r"S:\Computer Vision\Lab1\Part3\images\img6.png",
    cv2.IMREAD_GRAYSCALE
)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load images.")

img2 = cv2.resize(
    img2,
    (img1.shape[1], img1.shape[0])
)

img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)


# =========================================================
# HYBRID FUNCTION
# =========================================================

def create_hybrid(
    img1,
    img2,
    kernel_low,
    sigma_low,
    kernel_high,
    sigma_high,
    low_weight,
    high_weight
):

    # Low-pass
    low = cv2.GaussianBlur(
        img1,
        kernel_low,
        sigma_low
    )

    # High-pass
    blur2 = cv2.GaussianBlur(
        img2,
        kernel_high,
        sigma_high
    )

    high = img2 - blur2

    # Hybrid
    hybrid = (
        low_weight * low +
        high_weight * high
    )

    hybrid = np.clip(
        hybrid,
        0,
        255
    ).astype(np.uint8)

    return hybrid


# =========================================================
# EXPERIMENTS
# =========================================================

experiments = [

    (
        (11, 11), 3,
        (11, 11), 3,
        1.0, 1.0,
        "Exp 1"
    ),

    (
        (21, 21), 5,
        (21, 21), 5,
        1.0, 1.0,
        "Exp 2"
    ),

    (
        (31, 31), 8,
        (21, 21), 5,
        1.0, 1.0,
        "Exp 3"
    ),

    (
        (31, 31), 8,
        (21, 21), 5,
        1.2, 0.7,
        "Exp 4"
    ),

    (
        (31, 31), 8,
        (21, 21), 5,
        1.5, 0.5,
        "Exp 5"
    ),

    (
        (41, 41), 10,
        (21, 21), 5,
        1.5, 0.5,
        "Exp 6"
    )
]


# =========================================================
# ONE SCREEN
# =========================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)

fig.suptitle(
    "Hybrid Image Parameter Experiments",
    fontsize=18
)


# -------------------------
# ORIGINAL IMAGES
# -------------------------

axes[0, 0].imshow(
    img1,
    cmap="gray"
)

axes[0, 0].set_title(
    "Image 1\nLow-Pass Source"
)

axes[0, 0].axis("off")


axes[0, 1].imshow(
    img2,
    cmap="gray"
)

axes[0, 1].set_title(
    "Image 2\nHigh-Pass Source"
)

axes[0, 1].axis("off")


# -------------------------
# EXPERIMENTS
# -------------------------

for i, exp in enumerate(experiments):

    (
        kernel_low,
        sigma_low,
        kernel_high,
        sigma_high,
        low_weight,
        high_weight,
        name
    ) = exp

    hybrid = create_hybrid(
        img1,
        img2,
        kernel_low,
        sigma_low,
        kernel_high,
        sigma_high,
        low_weight,
        high_weight
    )

    # Start from column 2
    position = i + 2

    row = position // 4
    col = position % 4

    axes[row, col].imshow(
        hybrid,
        cmap="gray"
    )

    axes[row, col].set_title(
        f"{name}\n"
        f"Low K={kernel_low[0]}, σ={sigma_low}, W={low_weight}\n"
        f"High K={kernel_high[0]}, σ={sigma_high}, W={high_weight}"
    )

    axes[row, col].axis("off")


plt.tight_layout()
plt.show()
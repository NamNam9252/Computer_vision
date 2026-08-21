import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. READ IMAGES
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
    raise FileNotFoundError("Could not load one or both images.")


# Make both images the same size
img2 = cv2.resize(
    img2,
    (img1.shape[1], img1.shape[0])
)


# Convert to float
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)


# =========================================================
# 2. PARAMETERS
# =========================================================

levels = 5

# Bilateral filter parameters
d = 9
sigma_color = 75
sigma_space = 75


# =========================================================
# 3. BILATERAL PYRAMID
# =========================================================

def bilateral_pyramid(
    img,
    levels,
    d,
    sigma_color,
    sigma_space
):

    pyramid = [img]

    current = img.copy()

    for i in range(levels):

        # Bilateral filtering
        filtered = cv2.bilateralFilter(
            current.astype(np.float32),
            d,
            sigma_color,
            sigma_space
        )

        # Downsample
        current = cv2.resize(
            filtered,
            (
                filtered.shape[1] // 2,
                filtered.shape[0] // 2
            ),
            interpolation=cv2.INTER_LINEAR
        )

        pyramid.append(current)

    return pyramid


# =========================================================
# 4. NORMAL GAUSSIAN PYRAMID
# =========================================================

def gaussian_pyramid(img, levels):

    pyramid = [img]

    current = img.copy()

    for i in range(levels):

        current = cv2.pyrDown(current)

        pyramid.append(current)

    return pyramid


# =========================================================
# 5. LAPLACIAN PYRAMID
# =========================================================

def laplacian_pyramid(gaussian):

    laplacian = []

    for i in range(len(gaussian) - 1):

        h, w = gaussian[i].shape

        expanded = cv2.resize(
            gaussian[i + 1],
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        L = gaussian[i] - expanded

        laplacian.append(L)

    # Smallest level
    laplacian.append(
        gaussian[-1]
    )

    return laplacian


# =========================================================
# 6. CREATE PYRAMIDS FOR IMAGE 1
# =========================================================

gaussian1 = gaussian_pyramid(
    img1,
    levels
)

bilateral1 = bilateral_pyramid(
    img1,
    levels,
    d,
    sigma_color,
    sigma_space
)

laplacian1 = laplacian_pyramid(
    bilateral1
)


# =========================================================
# 7. CREATE PYRAMIDS FOR IMAGE 2
# =========================================================

gaussian2 = gaussian_pyramid(
    img2,
    levels
)

bilateral2 = bilateral_pyramid(
    img2,
    levels,
    d,
    sigma_color,
    sigma_space
)

laplacian2 = laplacian_pyramid(
    bilateral2
)


# =========================================================
# 8. MIX LAPLACIAN PYRAMIDS
# =========================================================

mixed_pyramid = []

for i in range(len(laplacian1)):

    # Fine levels → Image 2
    # Coarse levels → Image 1

    if i < 2:

        mixed = laplacian2[i]

    else:

        mixed = laplacian1[i]

    mixed_pyramid.append(
        mixed
    )


# =========================================================
# 9. RECONSTRUCT FINAL IMAGE
# =========================================================

result = mixed_pyramid[-1]

for i in range(
    len(mixed_pyramid) - 2,
    -1,
    -1
):

    h, w = mixed_pyramid[i].shape

    result = cv2.resize(
        result,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    result = result + mixed_pyramid[i]


result = np.clip(
    result,
    0,
    255
).astype(np.uint8)


# =========================================================
# 10. DISPLAY EVERYTHING ON ONE SCREEN
# =========================================================

plt.figure(figsize=(16, 10))


# ---------------------------------------------------------
# ORIGINAL IMAGE 1
# ---------------------------------------------------------

plt.subplot(3, 4, 1)

plt.imshow(
    img1,
    cmap="gray"
)

plt.title("Image 1")
plt.axis("off")


# ---------------------------------------------------------
# ORIGINAL IMAGE 2
# ---------------------------------------------------------

plt.subplot(3, 4, 2)

plt.imshow(
    img2,
    cmap="gray"
)

plt.title("Image 2")
plt.axis("off")


# ---------------------------------------------------------
# GAUSSIAN PYRAMID
# ---------------------------------------------------------

plt.subplot(3, 4, 3)

plt.imshow(
    gaussian1[2],
    cmap="gray"
)

plt.title("Gaussian Pyramid - Level 2")
plt.axis("off")


# ---------------------------------------------------------
# BILATERAL PYRAMID
# ---------------------------------------------------------

plt.subplot(3, 4, 4)

plt.imshow(
    bilateral1[2],
    cmap="gray"
)

plt.title("Bilateral Pyramid - Level 2")
plt.axis("off")


# ---------------------------------------------------------
# LAPLACIAN LEVEL
# ---------------------------------------------------------

lap_display = np.abs(
    laplacian1[1]
)

lap_display = cv2.normalize(
    lap_display,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


plt.subplot(3, 4, 5)

plt.imshow(
    lap_display,
    cmap="gray"
)

plt.title("Laplacian Level 1")
plt.axis("off")


# ---------------------------------------------------------
# BILATERAL LEVEL
# ---------------------------------------------------------

plt.subplot(3, 4, 6)

plt.imshow(
    bilateral1[1],
    cmap="gray"
)

plt.title("Bilateral Level 1")
plt.axis("off")


# ---------------------------------------------------------
# LOWEST GAUSSIAN
# ---------------------------------------------------------

plt.subplot(3, 4, 7)

plt.imshow(
    gaussian1[-1],
    cmap="gray"
)

plt.title("Smallest Gaussian Level")
plt.axis("off")


# ---------------------------------------------------------
# LOWEST BILATERAL
# ---------------------------------------------------------

plt.subplot(3, 4, 8)

plt.imshow(
    bilateral1[-1],
    cmap="gray"
)

plt.title("Smallest Bilateral Level")
plt.axis("off")


# ---------------------------------------------------------
# IMAGE 1 LAPLACIAN
# ---------------------------------------------------------

lap1_display = np.abs(
    laplacian1[0]
)

lap1_display = cv2.normalize(
    lap1_display,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


plt.subplot(3, 4, 9)

plt.imshow(
    lap1_display,
    cmap="gray"
)

plt.title("Laplacian Fine Details")
plt.axis("off")


# ---------------------------------------------------------
# IMAGE 2 LAPLACIAN
# ---------------------------------------------------------

lap2_display = np.abs(
    laplacian2[0]
)

lap2_display = cv2.normalize(
    lap2_display,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


plt.subplot(3, 4, 10)

plt.imshow(
    lap2_display,
    cmap="gray"
)

plt.title("Image 2 Fine Details")
plt.axis("off")


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

plt.subplot(3, 4, 11)

plt.imshow(
    result,
    cmap="gray"
)

plt.title("Bilateral + Laplacian Mixed")
plt.axis("off")


# ---------------------------------------------------------
# NORMAL GAUSSIAN MIX RESULT
# ---------------------------------------------------------

# Simple comparison
normal_mixed = (
    0.5 * img1 +
    0.5 * img2
)

normal_mixed = np.clip(
    normal_mixed,
    0,
    255
).astype(np.uint8)


plt.subplot(3, 4, 12)

plt.imshow(
    normal_mixed,
    cmap="gray"
)

plt.title("Simple Image Mix")
plt.axis("off")


plt.tight_layout()
plt.show()
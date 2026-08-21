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


# Make both images the same size
img2 = cv2.resize(
    img2,
    (img1.shape[1], img1.shape[0])
)


# Convert to float
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)


# =========================================================
# NUMBER OF PYRAMID LEVELS
# =========================================================

levels = 5


# =========================================================
# CREATE GAUSSIAN PYRAMID
# =========================================================

def gaussian_pyramid(img, levels):

    pyramid = [img]

    for i in range(levels):

        img = cv2.pyrDown(img)

        pyramid.append(img)

    return pyramid


# =========================================================
# CREATE LAPLACIAN PYRAMID
# =========================================================

def laplacian_pyramid(gaussian):

    laplacian = []

    for i in range(len(gaussian) - 1):

        # Size of current Gaussian level
        h, w = gaussian[i].shape

        # Expand next level
        expanded = cv2.pyrUp(
            gaussian[i + 1],
            dstsize=(w, h)
        )

        # Difference gives Laplacian
        L = gaussian[i] - expanded

        laplacian.append(L)

    # Last level is the smallest Gaussian image
    laplacian.append(
        gaussian[-1]
    )

    return laplacian


# =========================================================
# CREATE PYRAMIDS
# =========================================================

gaussian1 = gaussian_pyramid(
    img1,
    levels
)

gaussian2 = gaussian_pyramid(
    img2,
    levels
)


laplacian1 = laplacian_pyramid(
    gaussian1
)

laplacian2 = laplacian_pyramid(
    gaussian2
)


# =========================================================
# MIX PYRAMIDS
# =========================================================

mixed_pyramid = []


for i in range(len(laplacian1)):

    # Use Image 1 for large-scale information
    # Use Image 2 for fine details

    if i < 2:

        mixed = laplacian2[i]

    else:

        mixed = laplacian1[i]

    mixed_pyramid.append(
        mixed
    )


# =========================================================
# RECONSTRUCT IMAGE
# =========================================================

result = mixed_pyramid[-1]


for i in range(
    len(mixed_pyramid) - 2,
    -1,
    -1
):

    h, w = mixed_pyramid[i].shape

    result = cv2.pyrUp(
        result,
        dstsize=(w, h)
    )

    result = result + mixed_pyramid[i]


# Clip values
result = np.clip(
    result,
    0,
    255
).astype(np.uint8)


# =========================================================
# DISPLAY RESULTS
# =========================================================

plt.figure(figsize=(14, 8))


plt.subplot(2, 3, 1)

plt.imshow(
    img1,
    cmap="gray"
)

plt.title("Image 1")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    img2,
    cmap="gray"
)

plt.title("Image 2")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    result,
    cmap="gray"
)

plt.title("Pyramid Mixed Image")
plt.axis("off")


# Gaussian pyramid level
plt.subplot(2, 3, 4)

plt.imshow(
    gaussian1[2],
    cmap="gray"
)

plt.title("Gaussian Pyramid Level 2")
plt.axis("off")


# Laplacian pyramid
plt.subplot(2, 3, 5)

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

plt.imshow(
    lap_display,
    cmap="gray"
)

plt.title("Laplacian Pyramid Level 1")
plt.axis("off")


# Another Gaussian level
plt.subplot(2, 3, 6)

plt.imshow(
    gaussian1[-1],
    cmap="gray"
)

plt.title("Smallest Gaussian Level")
plt.axis("off")


plt.tight_layout()
plt.show()
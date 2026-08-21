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
# FOURIER TRANSFORM
# =========================================================

F1 = np.fft.fft2(img1)
F2 = np.fft.fft2(img2)

# Move low frequencies to the center
F1_shift = np.fft.fftshift(F1)
F2_shift = np.fft.fftshift(F2)


# =========================================================
# CREATE GAUSSIAN LOW-PASS FILTER
# =========================================================

h, w = img1.shape

cy = h // 2
cx = w // 2

Y, X = np.ogrid[:h, :w]

distance = np.sqrt(
    (X - cx) ** 2 +
    (Y - cy) ** 2
)


# Change this value to experiment
sigma = 40

low_pass_filter = np.exp(
    -(distance ** 2) /
    (2 * sigma ** 2)
)


# =========================================================
# HIGH-PASS FILTER
# =========================================================

high_pass_filter = 1 - low_pass_filter


# =========================================================
# APPLY FILTERS IN FREQUENCY DOMAIN
# =========================================================

low_frequency = F1_shift * low_pass_filter

high_frequency = F2_shift * high_pass_filter


# =========================================================
# CONVERT BACK TO SPATIAL DOMAIN
# =========================================================

low_image = np.fft.ifft2(
    np.fft.ifftshift(low_frequency)
)

high_image = np.fft.ifft2(
    np.fft.ifftshift(high_frequency)
)


# Fourier results are complex
# We only need the real part

low_image = np.real(low_image)

high_image = np.real(high_image)


# =========================================================
# CREATE HYBRID IMAGE
# =========================================================

hybrid = low_image + high_image

hybrid = np.clip(
    hybrid,
    0,
    255
).astype(np.uint8)


# =========================================================
# DISPLAY FREQUENCY SPECTRUMS
# =========================================================

spectrum1 = np.log(
    1 + np.abs(F1_shift)
)

spectrum2 = np.log(
    1 + np.abs(F2_shift)
)


# =========================================================
# DISPLAY EVERYTHING IN ONE SCREEN
# =========================================================

plt.figure(figsize=(16, 8))


# -------------------------
# IMAGE 1
# -------------------------

plt.subplot(2, 4, 1)

plt.imshow(
    img1,
    cmap="gray"
)

plt.title("Image 1")

plt.axis("off")


# -------------------------
# IMAGE 2
# -------------------------

plt.subplot(2, 4, 2)

plt.imshow(
    img2,
    cmap="gray"
)

plt.title("Image 2")

plt.axis("off")


# -------------------------
# FOURIER SPECTRUM 1
# -------------------------

plt.subplot(2, 4, 3)

plt.imshow(
    spectrum1,
    cmap="gray"
)

plt.title("Fourier Spectrum 1")

plt.axis("off")


# -------------------------
# FOURIER SPECTRUM 2
# -------------------------

plt.subplot(2, 4, 4)

plt.imshow(
    spectrum2,
    cmap="gray"
)

plt.title("Fourier Spectrum 2")

plt.axis("off")


# -------------------------
# LOW PASS FILTER
# -------------------------

plt.subplot(2, 4, 5)

plt.imshow(
    low_pass_filter,
    cmap="gray"
)

plt.title("Gaussian Low-Pass Filter")

plt.axis("off")


# -------------------------
# HIGH PASS FILTER
# -------------------------

plt.subplot(2, 4, 6)

plt.imshow(
    high_pass_filter,
    cmap="gray"
)

plt.title("Gaussian High-Pass Filter")

plt.axis("off")


# -------------------------
# LOW PASS IMAGE
# -------------------------

plt.subplot(2, 4, 7)

plt.imshow(
    low_image,
    cmap="gray"
)

plt.title("Low-Pass Image")

plt.axis("off")


# -------------------------
# HYBRID IMAGE
# -------------------------

plt.subplot(2, 4, 8)

plt.imshow(
    hybrid,
    cmap="gray"
)

plt.title("Hybrid Image")

plt.axis("off")


plt.tight_layout()

plt.show()
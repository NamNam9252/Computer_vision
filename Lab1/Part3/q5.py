import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


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


# Make same size
img2 = cv2.resize(
    img2,
    (img1.shape[1], img1.shape[0])
)


# Convert to float
img1 = img1.astype(np.float32)
img2 = img2.astype(np.float32)


# =========================================================
# PARAMETERS
# =========================================================

sigma = 40

low_weight = 1.0
high_weight = 1.0


# =========================================================
# SPATIAL DOMAIN
# =========================================================

start_spatial = time.perf_counter()


# Low-pass
low_spatial = cv2.GaussianBlur(
    img1,
    (31, 31),
    8
)


# High-pass
blur2_spatial = cv2.GaussianBlur(
    img2,
    (31, 31),
    8
)

high_spatial = img2 - blur2_spatial


# Hybrid
hybrid_spatial = (
    low_weight * low_spatial +
    high_weight * high_spatial
)

hybrid_spatial = np.clip(
    hybrid_spatial,
    0,
    255
).astype(np.uint8)


end_spatial = time.perf_counter()


spatial_time = end_spatial - start_spatial


# =========================================================
# FREQUENCY DOMAIN
# =========================================================

start_frequency = time.perf_counter()


# Fourier transform
F1 = np.fft.fft2(img1)
F2 = np.fft.fft2(img2)


# Shift zero frequency to center
F1_shift = np.fft.fftshift(F1)
F2_shift = np.fft.fftshift(F2)


# Create frequency coordinates
h, w = img1.shape

cy = h // 2
cx = w // 2

Y, X = np.ogrid[:h, :w]

distance = np.sqrt(
    (X - cx) ** 2 +
    (Y - cy) ** 2
)


# Gaussian low-pass filter
low_pass = np.exp(
    -(distance ** 2) /
    (2 * sigma ** 2)
)


# Gaussian high-pass
high_pass = 1 - low_pass


# Apply filters
low_frequency = F1_shift * low_pass
high_frequency = F2_shift * high_pass


# Inverse Fourier transform
low_frequency_image = np.fft.ifft2(
    np.fft.ifftshift(low_frequency)
)

high_frequency_image = np.fft.ifft2(
    np.fft.ifftshift(high_frequency)
)


# Take real part
low_frequency_image = np.real(
    low_frequency_image
)

high_frequency_image = np.real(
    high_frequency_image
)


# Hybrid
hybrid_frequency = (
    low_weight * low_frequency_image +
    high_weight * high_frequency_image
)

hybrid_frequency = np.clip(
    hybrid_frequency,
    0,
    255
).astype(np.uint8)


end_frequency = time.perf_counter()


frequency_time = end_frequency - start_frequency


# =========================================================
# EXECUTION TIME
# =========================================================

print("------------------------------------")
print("Execution Time")
print("------------------------------------")

print(
    f"Spatial Domain   : "
    f"{spatial_time * 1000:.4f} ms"
)

print(
    f"Frequency Domain : "
    f"{frequency_time * 1000:.4f} ms"
)


if spatial_time < frequency_time:

    print("\nSpatial domain is faster.")

else:

    print("\nFrequency domain is faster.")


# =========================================================
# VISUAL COMPARISON
# =========================================================

plt.figure(figsize=(12, 6))


plt.subplot(1, 3, 1)

plt.imshow(
    img1,
    cmap="gray"
)

plt.title("Image 1")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    hybrid_spatial,
    cmap="gray"
)

plt.title(
    f"Spatial Domain\n"
    f"{spatial_time * 1000:.2f} ms"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    hybrid_frequency,
    cmap="gray"
)

plt.title(
    f"Frequency Domain\n"
    f"{frequency_time * 1000:.2f} ms"
)

plt.axis("off")


plt.tight_layout()
plt.show()
import cv2
import numpy as np
from scipy.signal import convolve2d
import os, matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    mask  = np.random.random(img.shape)
    noisy[mask < prob / 2]      = 0
    noisy[mask > 1 - prob / 2]  = 255
    return noisy

def add_gaussian_noise(img, sigma=25):
    return np.clip(img.astype(np.float64) + np.random.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)

def weighted_avg_kernel(k):
    h = np.arange(1, k // 2 + 2)
    w1d = np.concatenate([h, h[-2::-1]])
    k2d = np.outer(w1d, w1d).astype(np.float64)
    return k2d / k2d.sum()

def box_filter_manual(img, k):
    """Box filter using numpy convolution — no cv2 built-in."""
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    out = convolve2d(img.astype(np.float64), kernel, mode="same", boundary="symm")
    return np.clip(out, 0, 255).astype(np.uint8)

K = 5
img1 = cv2.imread("images/img1.gif", cv2.IMREAD_GRAYSCALE)  # fine details
img2 = cv2.imread("images/img2.gif", cv2.IMREAD_GRAYSCALE)  # smooth regions

noisy1 = add_salt_pepper(img1)
noisy2 = add_gaussian_noise(img2)

cv2.imwrite("output/noisy1_sp.png",     noisy1)
cv2.imwrite("output/noisy2_gauss.png",  noisy2)

for noisy, tag in [(noisy1, "sp"), (noisy2, "gauss")]:
    cv2.imwrite(f"output/{tag}_box.png",      box_filter_manual(noisy, K))
    cv2.imwrite(f"output/{tag}_wavg.png",     cv2.filter2D(noisy, -1, weighted_avg_kernel(K)))
    cv2.imwrite(f"output/{tag}_gaussian.png", cv2.GaussianBlur(noisy, (K, K), 0))
    cv2.imwrite(f"output/{tag}_median.png",   cv2.medianBlur(noisy, K))

print("q1 done — noisy images and filtered outputs saved.")

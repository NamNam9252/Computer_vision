import cv2
import numpy as np
from scipy.signal import convolve2d


sizes = [3, 5, 7]

img1 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img3.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img4.png", cv2.IMREAD_GRAYSCALE)

imgns1 = img1 

imgns2 = img2 
# Weighted-average kernels
k3 = np.ones((3,3), np.float32) / 9
k5 = np.ones((5,5), np.float32) / 25
k7 = np.ones((7,7), np.float32) / 49

kernels = [k3, k5, k7]

# -------- Image 1: Salt & Pepper --------
box1 = [cv2.blur(imgns1, (k,k)) for k in sizes]
weighted1 = [cv2.filter2D(imgns1, -1, k) for k in kernels]
gaussian1 = [cv2.GaussianBlur(imgns1, (k,k), 0) for k in sizes]
median1 = [cv2.medianBlur(imgns1, k) for k in sizes]

# -------- Image 2: Gaussian Noise --------
box2 = [cv2.blur(imgns2, (k,k)) for k in sizes]
weighted2 = [cv2.filter2D(imgns2, -1, k) for k in kernels]
gaussian2 = [cv2.GaussianBlur(imgns2, (k,k), 0) for k in sizes]
median2 = [cv2.medianBlur(imgns2, k) for k in sizes]




def mse(original, filtered):
    return np.mean((original.astype(np.float64) - filtered.astype(np.float64))**2)


print("\nComparison of Image 1 by MSE")

print("Box 3x3:", mse(img1, box1[0]))
print("Box 5x5:", mse(img1, box1[1]))
print("Box 7x7:", mse(img1, box1[2]))

print("Weighted 3x3:", mse(img1, weighted1[0]))
print("Weighted 5x5:", mse(img1, weighted1[1]))
print("Weighted 7x7:", mse(img1, weighted1[2]))

print("Gaussian 3x3:", mse(img1, gaussian1[0]))
print("Gaussian 5x5:", mse(img1, gaussian1[1]))
print("Gaussian 7x7:", mse(img1, gaussian1[2]))

print("Median 3x3:", mse(img1, median1[0]))
print("Median 5x5:", mse(img1, median1[1]))
print("Median 7x7:", mse(img1, median1[2]))


print("\nComparison of Image 2 by MSE")

print("Box 3x3:", mse(img2, box2[0]))
print("Box 5x5:", mse(img2, box2[1]))
print("Box 7x7:", mse(img2, box2[2]))

print("Weighted 3x3:", mse(img2, weighted2[0]))
print("Weighted 5x5:", mse(img2, weighted2[1]))
print("Weighted 7x7:", mse(img2, weighted2[2]))

print("Gaussian 3x3:", mse(img2, gaussian2[0]))
print("Gaussian 5x5:", mse(img2, gaussian2[1]))
print("Gaussian 7x7:", mse(img2, gaussian2[2]))

print("Median 3x3:", mse(img2, median2[0]))
print("Median 5x5:", mse(img2, median2[1]))
print("Median 7x7:", mse(img2, median2[2]))


# PSNR

def psnr(original, filtered):
    return 10 * np.log10(255**2 / mse(original, filtered))


print("\nComparison of Image 1 by PSNR")

print("Box 3x3:", psnr(img1, box1[0]))
print("Box 5x5:", psnr(img1, box1[1]))
print("Box 7x7:", psnr(img1, box1[2]))

print("Weighted 3x3:", psnr(img1, weighted1[0]))
print("Weighted 5x5:", psnr(img1, weighted1[1]))
print("Weighted 7x7:", psnr(img1, weighted1[2]))

print("Gaussian 3x3:", psnr(img1, gaussian1[0]))
print("Gaussian 5x5:", psnr(img1, gaussian1[1]))
print("Gaussian 7x7:", psnr(img1, gaussian1[2]))

print("Median 3x3:", psnr(img1, median1[0]))
print("Median 5x5:", psnr(img1, median1[1]))
print("Median 7x7:", psnr(img1, median1[2]))


print("\nComparison of Image 2 by PSNR")

print("Box 3x3:", psnr(img2, box2[0]))
print("Box 5x5:", psnr(img2, box2[1]))
print("Box 7x7:", psnr(img2, box2[2]))

print("Weighted 3x3:", psnr(img2, weighted2[0]))
print("Weighted 5x5:", psnr(img2, weighted2[1]))
print("Weighted 7x7:", psnr(img2, weighted2[2]))

print("Gaussian 3x3:", psnr(img2, gaussian2[0]))
print("Gaussian 5x5:", psnr(img2, gaussian2[1]))
print("Gaussian 7x7:", psnr(img2, gaussian2[2]))

print("Median 3x3:", psnr(img2, median2[0]))
print("Median 5x5:", psnr(img2, median2[1]))
print("Median 7x7:", psnr(img2, median2[2]))
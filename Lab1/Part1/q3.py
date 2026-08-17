import cv2 
import numpy as np 

def mse(original, reconstructed):
    return np.mean(
        (original.astype(np.float64) -
         reconstructed.astype(np.float64)) ** 2
    )

def psnr(original,reconstructed):
    return 10*np.log10(255**2/mse(original,reconstructed))

img512 = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\images\\img1.gif")

img512_128 = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\output\\img_128_nearest.png")

img512_256 = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\output\\img_256_nearest.png")

print("Error of 128 = " , mse(np.array(img512),np.array(img512_128)) , "Error of 256 = ",mse(np.array(img512),np.array(img512_256)))





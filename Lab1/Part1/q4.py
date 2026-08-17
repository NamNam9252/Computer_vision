# 4. error maps

import matplotlib.pyplot as plt
import cv2
import numpy as np 

def error_map(original, reconstructed):
    absolute_error = np.abs(
        original.astype(np.float64) -
        reconstructed.astype(np.float64)
    )

    squared_error = (
        original.astype(np.float64) -
        reconstructed.astype(np.float64)
    ) ** 2

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(absolute_error, cmap="gray")
    plt.title("Absolute Error")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(squared_error, cmap="gray")
    plt.title("Squared Error")
    plt.axis("off")

    plt.show()




img = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\images\\img1.gif")

img128 = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\output\\img_128_nearest.png")

img256 = cv2.imread(f"S:\\Computer Vision\\Lab1\\Part1\\output\\img_256_nearest.png")

error_map(img, img128)
error_map(img ,img256)

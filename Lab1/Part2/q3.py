import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from utils import filters

base_dir = Path(__file__).resolve().parent
images_dir = base_dir / "images"

imgns1 = cv2.imread(str(images_dir / "img3.png"), cv2.IMREAD_GRAYSCALE)
imgns2 = cv2.imread(str(images_dir / "img4.png"), cv2.IMREAD_GRAYSCALE)

img1_clean = cv2.imread(str(images_dir / "img1.png"), cv2.IMREAD_GRAYSCALE)
img2_clean = cv2.imread(str(images_dir / "img2.png"), cv2.IMREAD_GRAYSCALE)

kernel_weighted = np.array([[1,2,1],
                             [2,4,2],
                             [1,2,1]], dtype=np.float32) / 16

def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / m)


filter_names = ["Box", "Weighted Avg", "Gaussian", "Median"]

filtered_ns1 = [
    filters.boxFilter(imgns1, 3),
    filters.weightedAverageFilter(imgns1, kernel_weighted),
    filters.gaussianFilter(imgns1, 3, 1),
    filters.medianFilter(imgns1, 3),
]

filtered_ns2 = [
    filters.boxFilter(imgns2, 3),
    filters.weightedAverageFilter(imgns2, kernel_weighted),
    filters.gaussianFilter(imgns2, 3, 1),
    filters.medianFilter(imgns2, 3),
]

records = []

for name, f1, f2 in zip(filter_names, filtered_ns1, filtered_ns2):
    records.append({
        "Image":           "Salt & Pepper",
        "Filter":          name,
        "MSE":             round(mse(img1_clean, f1), 4),
        "PSNR (dB)":       round(psnr(img1_clean, f1), 4),
    })
    records.append({
        "Image":           "Gaussian Noise",
        "Filter":          name,
        "MSE":             round(mse(img2_clean, f2), 4),
        "PSNR (dB)":       round(psnr(img2_clean, f2), 4),
    })

df = pd.DataFrame(records)

df_snp = df[df["Image"] == "Salt & Pepper"].drop(columns="Image").sort_values("PSNR (dB)", ascending=False).reset_index(drop=True)
df_gau = df[df["Image"] == "Gaussian Noise"].drop(columns="Image").sort_values("PSNR (dB)", ascending=False).reset_index(drop=True)

print("===== Image 1: Salt & Pepper Noise =====")
print(df_snp.to_string(index=False))

print("\n===== Image 2: Gaussian Noise =====")
print(df_gau.to_string(index=False))

print(f"\nBest for Salt & Pepper : {df_snp.iloc[0]['Filter']}  -> PSNR={df_snp.iloc[0]['PSNR (dB)']} dB")
print(f"Best for Gaussian Noise: {df_gau.iloc[0]['Filter']}  -> PSNR={df_gau.iloc[0]['PSNR (dB)']} dB")

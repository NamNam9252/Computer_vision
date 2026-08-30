import cv2
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
img_path = base_dir / "images" / "img1.gif"
output_dir = base_dir / "output"

original = cv2.imread(str(img_path))
original = cv2.resize(original, (512, 512))

def mse(a, b):
    return np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / m)

def edge_strength(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))

scales = [256, 128]
methods = {
    "Nearest-Neighbour": cv2.INTER_NEAREST,
    "Bilinear":          cv2.INTER_LINEAR,
    "Bicubic":           cv2.INTER_CUBIC,
}

records = []

for scale in scales:
    downsampled = cv2.resize(original, (scale, scale), interpolation=cv2.INTER_AREA)
    for method_name, flag in methods.items():
        recon = cv2.resize(downsampled, (512, 512), interpolation=flag)
        records.append({
            "Scale":         f"{scale}x{scale}",
            "Method":        method_name,
            "MSE":           round(mse(original, recon), 4),
            "PSNR (dB)":     round(psnr(original, recon), 4),
        })

df = pd.DataFrame(records)
df = df.sort_values("PSNR(dB)", ascending=False).reset_index(drop=True)

print(df.to_string(index=False))
print(f"\nBest: {df.iloc[0]['Method']} at {df.iloc[0]['Scale']} -> PSNR={df.iloc[0]['PSNR (dB)']} dB")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

original_edge = edge_strength(original)

records = []
reconstructed_images = {}

for scale in scales:
    downsampled = cv2.resize(original, (scale, scale), interpolation=cv2.INTER_AREA)
    for method_name, flag in methods.items():
        recon = cv2.resize(downsampled, (512, 512), interpolation=flag)
        m = mse(original, recon)
        p = psnr(original, recon)
        e = edge_strength(recon)
        records.append({
            "Scale":         f"{scale}x{scale}",
            "Method":        method_name,
            "MSE":           round(m, 4),
            "PSNR (dB)":     round(p, 4),
            "Edge Strength": round(e, 4),
        })
        reconstructed_images[(scale, method_name)] = recon

df = pd.DataFrame(records)
df["Rank (PSNR)"] = df["PSNR (dB)"].rank(ascending=False).astype(int)
df = df.sort_values("PSNR (dB)", ascending=False).reset_index(drop=True)

print("\n===== Interpolation Comparison Table =====\n")
print(df.to_string(index=False))

best = df.iloc[0]
print(f"\nBest Method: {best['Method']} at {best['Scale']}")
print(f"  MSE  = {best['MSE']}")
print(f"  PSNR = {best['PSNR (dB)']} dB")

fig, axes = plt.subplots(len(scales), len(methods) + 1, figsize=(18, 7))

for row, scale in enumerate(scales):
    axes[row, 0].imshow(original, cmap="gray")
    axes[row, 0].set_title("Original 512x512")
    axes[row, 0].axis("off")

    for col, method_name in enumerate(methods):
        recon = reconstructed_images[(scale, method_name)]
        m = mse(original, recon)
        p = psnr(original, recon)
        axes[row, col + 1].imshow(recon, cmap="gray")
        axes[row, col + 1].set_title(
            f"{method_name}\n({scale}x{scale}->512)\nMSE={m:.2f} | PSNR={p:.2f}dB",
            fontsize=8
        )
        axes[row, col + 1].axis("off")

plt.suptitle("Q6 - Interpolation Method Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(str(output_dir / "q6_comparison.png"), dpi=150)
plt.show()

fig2, axes2 = plt.subplots(len(scales), len(methods), figsize=(15, 7))

for row, scale in enumerate(scales):
    for col, method_name in enumerate(methods):
        recon = reconstructed_images[(scale, method_name)]
        abs_err = np.abs(original.astype(np.float64) - recon.astype(np.float64))
        im = axes2[row, col].imshow(abs_err, cmap="hot")
        axes2[row, col].set_title(
            f"{method_name} ({scale}->512)\nMax={abs_err.max():.1f}", fontsize=8
        )
        axes2[row, col].axis("off")
        plt.colorbar(im, ax=axes2[row, col], fraction=0.046, pad=0.04)

plt.suptitle("Q6 - Absolute Error Maps", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(str(output_dir / "q6_error_maps.png"), dpi=150)
plt.show()

print("\nSaved: output/q6_comparison.png")
print("Saved: output/q6_error_maps.png")

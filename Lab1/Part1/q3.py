import cv2
import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────

def mse(original, reconstructed):
    """Mean Squared Error between two images (lower is better)."""
    return np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)

def psnr(original, reconstructed):
    """Peak Signal-to-Noise Ratio in dB (higher is better).
       Returns inf when MSE == 0 (identical images).
    """
    error = mse(original, reconstructed)
    if error == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(error))

# ── load original and resize to the comparison size (512x512) ──────────────

TARGET_SIZE = (512, 512)

original = cv2.imread("images/img1.gif")
original_512 = cv2.resize(original, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

# ── reconstructed images produced by Q2 ───────────────────────────────────
# Naming convention from q2.py:
#   img{source_idx}_{method_idx}.png
#   source 0 -> downscaled from 128x128
#   source 1 -> downscaled from 256x256
#   method 0 -> INTER_NEAREST
#   method 1 -> INTER_LINEAR
#   method 2 -> INTER_CUBIC

source_labels  = ["128->512", "256->512"]
method_labels  = ["Nearest", "Linear", "Cubic"]

# ── compute and display metrics ────────────────────────────────────────────

print("=" * 65)
print(f"{'Image':<22} {'Source':<10} {'Method':<10} {'MSE':>10} {'PSNR (dB)':>12}")
print("=" * 65)

results = []

for src_idx, src_label in enumerate(source_labels):
    for mth_idx, mth_label in enumerate(method_labels):
        path = f"output/img{src_idx}_{mth_idx}.png"
        img  = cv2.imread(path)

        if img is None:
            print(f"  [WARNING] Could not load {path} - skipping.")
            continue

        m = mse(original_512, img)
        p = psnr(original_512, img)

        filename = f"img{src_idx}_{mth_idx}.png"
        print(f"{filename:<22} {src_label:<10} {mth_label:<10} {m:>10.4f} {p:>12.4f}")

        results.append({
            "file":   filename,
            "source": src_label,
            "method": mth_label,
            "mse":    m,
            "psnr":   p,
        })

print("=" * 65)

# ── find best and worst reconstructions ───────────────────────────────────

best  = min(results, key=lambda r: r["mse"])
worst = max(results, key=lambda r: r["mse"])

print(f"\nBest  reconstruction (lowest MSE) : {best['file']}"
      f"  [Source: {best['source']}, Method: {best['method']}]"
      f"  MSE={best['mse']:.4f}  PSNR={best['psnr']:.4f} dB")

print(f"Worst reconstruction (highest MSE): {worst['file']}"
      f"  [Source: {worst['source']}, Method: {worst['method']}]"
      f"  MSE={worst['mse']:.4f}  PSNR={worst['psnr']:.4f} dB")

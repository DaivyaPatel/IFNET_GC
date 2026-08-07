#!/usr/bin/env python3
"""
pysteps_eagle_test.py
=====================
Kaggle/Eagle standalone test for pysteps optical-flow interpolation.
Hardcoded inputs: T0, T1 → interpolates midpoint T0.5.

Optional: Provide GT_TIR and GT_WV for the midpoint to compute metrics.

Outputs (saved to /kaggle/working/pysteps_output/):
  1. interpolated.png          — the pysteps midpoint frame
  2. comparison.png            — side-by-side: T0 | Interp | T1
  3. animation.gif             — 3-frame looping GIF
  4. flow_visualization.png    — HSV-encoded optical flow field
  5. metrics.txt               — RMSE, PSNR, SSIM, FSIM (if GT provided)
"""

import os
import sys
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

from pysteps import motion
from pysteps.extrapolation.semilagrangian import extrapolate
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# ==============================================================
#  INPUTS — edit these
# ==============================================================

# T0 and T1 (required)
DEFAULT_TIR0 = "/kaggle/input/datasets/krishs23/milton-test/test-milton/OR_ABI-L2-CMIPF-M6C13_G16_s20242811800205_e20242811809525_c20242811812316_crop256.nc"
DEFAULT_TIR1 = "/kaggle/input/datasets/krishs23/milton-test/test-milton/OR_ABI-L2-CMIPF-M6C13_G16_s20242811820205_e20242811829525_c20242811829596_crop256.nc"
DEFAULT_WV0  = "/kaggle/input/datasets/krishs23/milton-test/test-milton/OR_ABI-L2-CMIPF-M6C08_G16_s20242811800205_e20242811809513_c20242811809587_crop256.nc"
DEFAULT_WV1  = "/kaggle/input/datasets/krishs23/milton-test/test-milton/OR_ABI-L2-CMIPF-M6C08_G16_s20242811820205_e20242811829513_c20242811829593_crop256.nc"

# Ground truth at midpoint (optional — set to None if you don't have it)
# Example path pattern for T0.5 (10 min after T0):
DEFAULT_GT_TIR = "/kaggle/input/datasets/krishs23/milton-test/test-milton/Real/OR_ABI-L2-CMIPF-M6C13_G16_s20242811810205_e20242811819525_c20242811819587_crop256.nc"   # "/kaggle/input/.../OR_ABI-L2-CMIPF-M6C13_G16_s20242811810205_....nc"
DEFAULT_GT_WV  = "/kaggle/input/datasets/krishs23/milton-test/test-milton/Real/OR_ABI-L2-CMIPF-M6C08_G16_s20242811810205_e20242811819513_c20242811819590_crop256.nc"   # "/kaggle/input/.../OR_ABI-L2-CMIPF-M6C08_G16_s20242811810205_....nc"

OUTPUT_DIR = "/kaggle/working/pysteps_output"

GIF_FPS   = 2
GIF_SCALE = 1.0

BT13_MIN, BT13_MAX = 190.0, 310.0
BT8_MIN,  BT8_MAX  = 190.0, 280.0

PYSTEPS_METHOD = "lucaskanade"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
#  HELPERS
# ==============================================================

def load_nc(path):
    ds = xr.open_dataset(path)
    bt = ds["CMI"].values.astype(np.float32)
    if "DQF" in ds:
        bt = np.where(ds["DQF"].values <= 1, bt, np.nan)
    ds.close()
    return bt

def normalize_bt(bt, vmin, vmax):
    bt = np.clip(bt, vmin, vmax)
    return (vmax - bt) / (vmax - vmin)

def make_rgb(bt13, bt8):
    r = normalize_bt(bt13, BT13_MIN, BT13_MAX)
    g = normalize_bt(bt8,  BT8_MIN,  BT8_MAX)
    b = (r + g) / 2
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

def save_png(arr_01, path):
    Image.fromarray((arr_01 * 255).astype(np.uint8)).save(path)

# ==============================================================
#  PYSTEPS INTERPOLATION
# ==============================================================

def pysteps_interpolate_channel(field0, field1, frac=0.5, method="lucaskanade"):
    """Bidirectional semi-Lagrangian advection at fraction `frac`."""
    f0 = np.nan_to_num(field0, nan=np.nanmean(field0))
    f1 = np.nan_to_num(field1, nan=np.nanmean(field1))

    oflow = motion.get_method(method)
    stack = np.stack([f0, f1], axis=0).astype(np.float64)
    velocity = oflow(stack)

    fwd = extrapolate(f0, velocity * frac, 1)[-1]
    bwd = extrapolate(f1, -velocity * (1.0 - frac), 1)[-1]

    interp = (1.0 - frac) * fwd + frac * bwd
    return interp.astype(np.float32), velocity

def run_pysteps(bt13_a, bt8_a, bt13_b, bt8_b, frac=0.5):
    tir_interp, tir_flow = pysteps_interpolate_channel(bt13_a, bt13_b, frac, PYSTEPS_METHOD)
    wv_interp,  wv_flow  = pysteps_interpolate_channel(bt8_a,  bt8_b,  frac, PYSTEPS_METHOD)
    rgb = make_rgb(tir_interp, wv_interp)
    return rgb, tir_interp, wv_interp, tir_flow, wv_flow

# ==============================================================
#  METRICS
# ==============================================================

def compute_metrics(pred_bt13, pred_bt8, gt_bt13, gt_bt8):
    out = {}
    for ch, pred, gt, vmin, vmax in [
        ("CH13", pred_bt13, gt_bt13, BT13_MIN, BT13_MAX),
        ("CH8",  pred_bt8,  gt_bt8,  BT8_MIN,  BT8_MAX),
    ]:
        valid = ~(np.isnan(pred) | np.isnan(gt))
        if not valid.any():
            out[f"RMSE_{ch}"] = out[f"PSNR_{ch}"] = out[f"SSIM_{ch}"] = np.nan
            continue
        p = np.where(valid, pred.astype(np.float64), gt.astype(np.float64))
        g = gt.astype(np.float64)
        dr = float(vmax - vmin)
        out[f"RMSE_{ch}"] = float(np.sqrt(np.mean((p - g) ** 2)))
        pn = np.clip((p - vmin) / dr, 0, 1)
        gn = np.clip((g - vmin) / dr, 0, 1)
        out[f"PSNR_{ch}"] = float(psnr_fn(gn, pn, data_range=1.0))
        out[f"SSIM_{ch}"] = float(ssim_fn(gn, pn, data_range=1.0))

    out["RMSE_avg"] = float(np.nanmean([out.get("RMSE_CH13", np.nan), out.get("RMSE_CH8", np.nan)]))
    out["PSNR_avg"] = float(np.nanmean([out.get("PSNR_CH13", np.nan), out.get("PSNR_CH8", np.nan)]))
    out["SSIM_avg"] = float(np.nanmean([out.get("SSIM_CH13", np.nan), out.get("SSIM_CH8", np.nan)]))

    try:
        import piq
        import torch
        def _norm(arr, vmin, vmax):
            mid = (vmin + vmax) / 2.0
            safe = np.where(np.isnan(arr), mid, arr.astype(np.float64))
            return np.clip((safe - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        def _t(ch13, ch8):
            r = _norm(ch13, BT13_MIN, BT13_MAX)
            g = _norm(ch8,  BT8_MIN,  BT8_MAX)
            b = (r + g) / 2
            return torch.from_numpy(np.stack([r, g, b])).unsqueeze(0)
        out["FSIM"] = float(piq.fsim(_t(pred_bt13, pred_bt8), _t(gt_bt13, gt_bt8), data_range=1.0))
    except Exception:
        out["FSIM"] = float("nan")
    return out

# ==============================================================
#  FLOW VISUALIZATION
# ==============================================================

def flow_to_hsv(flow):
    u, v = flow[0], flow[1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v_norm = mag / (mag.max() + 1e-8)
    import matplotlib.colors as mcolors
    hsv = np.stack([h, s, v_norm], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)

# ==============================================================
#  MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("Pysteps Eagle Test — Single Midpoint Interpolation")
    print("=" * 60)

    # --- Load inputs ---
    print("\nLoading GOES .nc files...")
    bt13_t0 = load_nc(DEFAULT_TIR0)
    bt13_t1 = load_nc(DEFAULT_TIR1)
    bt8_t0  = load_nc(DEFAULT_WV0)
    bt8_t1  = load_nc(DEFAULT_WV1)

    rgb_t0 = make_rgb(bt13_t0, bt8_t0)
    rgb_t1 = make_rgb(bt13_t1, bt8_t1)

    print(f"  T0 shape: {bt13_t0.shape}")
    print(f"  T1 shape: {bt13_t1.shape}")

    # --- Interpolate ---
    print(f"\nRunning pysteps interpolation (method={PYSTEPS_METHOD}, frac=0.5)...")
    rgb_interp, tir_interp, wv_interp, tir_flow, wv_flow = run_pysteps(
        bt13_t0, bt8_t0, bt13_t1, bt8_t1, frac=0.5
    )
    print("  Interpolation complete.")

    # --- Save outputs ---
    save_png(rgb_interp, os.path.join(OUTPUT_DIR, "interpolated.png"))

    flow_hsv = flow_to_hsv(tir_flow)
    Image.fromarray(flow_hsv).save(os.path.join(OUTPUT_DIR, "flow_visualization.png"))

    # --- Metrics (if ground truth provided) ---
    metrics = None
    if DEFAULT_GT_TIR and DEFAULT_GT_WV and os.path.exists(DEFAULT_GT_TIR) and os.path.exists(DEFAULT_GT_WV):
        print("\nGround truth found — computing metrics...")
        bt13_gt = load_nc(DEFAULT_GT_TIR)
        bt8_gt  = load_nc(DEFAULT_GT_WV)
        metrics = compute_metrics(tir_interp, wv_interp, bt13_gt, bt8_gt)

        print(f"\n{'='*60}")
        print("METRICS (Interpolated vs Ground Truth)")
        print(f"{'='*60}")
        print(f"  RMSE avg : {metrics['RMSE_avg']:.3f} K")
        print(f"  PSNR avg : {metrics['PSNR_avg']:.2f} dB")
        print(f"  SSIM avg : {metrics['SSIM_avg']:.4f}")
        print(f"  FSIM     : {metrics['FSIM']:.4f}")
        print(f"{'='*60}")

        # Save to file
        with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
            f.write("Pysteps Interpolation Metrics\n")
            f.write("=" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k:<12} : {v:.4f}\n")
        print("  metrics.txt saved.")
    else:
        print("\n[NOTE] No ground-truth files provided. Metrics skipped.")
        print("       Set DEFAULT_GT_TIR and DEFAULT_GT_WV to enable RMSE/PSNR/SSIM/FSIM.")

    # --- Side-by-side comparison ---
    print("\nBuilding comparison PNG...")
    h, w = rgb_t0.shape[:2]
    GAP = 8
    HDR = 44
    canvas = np.full((h + HDR, w * 3 + GAP * 4, 3), 18, dtype=np.uint8)

    canvas[HDR:HDR+h, GAP:GAP+w]                             = (rgb_t0     * 255).astype(np.uint8)
    canvas[HDR:HDR+h, GAP+w+GAP:GAP+w+GAP+w]                 = (rgb_interp * 255).astype(np.uint8)
    canvas[HDR:HDR+h, GAP+w+GAP+w+GAP:GAP+w+GAP+w+GAP+w]   = (rgb_t1     * 255).astype(np.uint8)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, HDR - 1], fill=(10, 10, 22))

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    draw.text((GAP + w // 2, HDR // 2), "T0  REAL", fill=(80, 210, 120), font=font, anchor="mm")
    draw.text((GAP + w + GAP + w // 2, HDR // 2), "T0.5  PYSTEPS", fill=(255, 200, 80), font=font, anchor="mm")
    draw.text((GAP + w + GAP + w + GAP + w // 2, HDR // 2), "T1  REAL", fill=(90, 160, 255), font=font, anchor="mm")

    pil.save(os.path.join(OUTPUT_DIR, "comparison.png"))
    print("  comparison.png saved.")

    # --- 3-frame looping GIF ---
    print("\nBuilding animation GIF...")
    duration_ms = int(1000 / GIF_FPS)

    def make_gif_frame(img, label, frame_num, total):
        H, W = img.shape[:2]
        hdr = 36
        c = np.full((H + hdr, W, 3), 18, dtype=np.uint8)
        c[hdr:hdr+H, :] = (img * 255).astype(np.uint8)
        p = Image.fromarray(c)
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, W, hdr - 1], fill=(10, 10, 22))
        try:
            f = ImageFont.truetype("arial.ttf", 13)
            fsm = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            f = fsm = ImageFont.load_default()
        d.text((W // 2, hdr // 2), label, fill=(200, 200, 200), font=f, anchor="mm")
        d.text((W - 4, hdr - 4), f"{frame_num}/{total}", fill=(130, 130, 130), font=fsm, anchor="rb")
        return p

    frames = [
        make_gif_frame(rgb_t0,     "T0  REAL",     1, 3),
        make_gif_frame(rgb_interp, "T0.5  PYSTEPS", 2, 3),
        make_gif_frame(rgb_t1,     "T1  REAL",     3, 3),
    ]

    if GIF_SCALE != 1.0:
        frames = [f.resize((int(f.width * GIF_SCALE), int(f.height * GIF_SCALE)), Image.LANCZOS) for f in frames]

    gif_path = os.path.join(OUTPUT_DIR, "animation.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  animation.gif saved  ({os.path.getsize(gif_path)/1e6:.1f} MB)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    print(f"  Output folder: {OUTPUT_DIR}")
    print(f"    interpolated.png       — midpoint RGB frame")
    print(f"    comparison.png         — T0 | Interp | T1")
    print(f"    animation.gif          — 3-frame looping GIF")
    print(f"    flow_visualization.png — HSV optical flow (TIR channel)")
    if metrics:
        print(f"    metrics.txt            — RMSE, PSNR, SSIM, FSIM")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
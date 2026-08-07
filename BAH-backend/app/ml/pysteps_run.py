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
# pyrefly: ignore [missing-import]
import piq
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

try:
    from pysteps import motion
    from pysteps.extrapolation.semilagrangian import extrapolate
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

BT13_MIN, BT13_MAX = 190.0, 310.0
BT8_MIN,  BT8_MAX  = 190.0, 280.0
PYSTEPS_METHOD = "lucaskanade"
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
    b = normalize_bt(bt8,  BT8_MIN,  BT8_MAX)
    g = (r + b) / 2   
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

def save_png(arr_01, path):
    Image.fromarray((arr_01 * 255).astype(np.uint8)).save(path)

# ==============================================================
#  PYSTEPS INTERPOLATION
# ==============================================================

def pysteps_interpolate_channel(field0, field1, frac=0.5, method="lucaskanade"):
    """Bidirectional semi-Lagrangian advection at fraction `frac`."""
    if not PYSTEPS_AVAILABLE:
        raise NotImplementedError("pysteps is not installed")
        
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
        # import piq
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

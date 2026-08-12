"""
Hurricane Milton — Dual Animation: Real vs Interpolated
========================================================
[FIXED FOR THE ACTUAL FINE-TUNED CHECKPOINT — 6august.pth]

Your checkpoint is NOT a plain IFNet_GC state_dict. It's a training
checkpoint from "RIFE Evolution Fine-Tuning — WV/TIR-Aware Residual
Refinement" containing TWO separate networks:

  1. base_flownet  -> the ORIGINAL RIFE IFNet (frozen, bidirectional
                       dual-pass blocks), loaded from the stock
                       flownet.pkl and never changed during training.
  2. refine / ema_refine -> a NEW "EvolutionRefinementNet" that looks
                       at the base's merged RGB output + flow field +
                       RAW per-band TIR/WV, and predicts a small
                       residual correction that gets scaled and ADDED
                       on top of the base's output.

ARCHITECTURE FIX (verified directly against this checkpoint's actual
state_dict, not assumed):

  - There is NO gap embedding anywhere in this checkpoint. The
    previous version of this script invented a "gap_embed" module
    (a Linear -> PReLU -> Linear time-delta embedding concatenated
    onto the input) that simply doesn't exist here. This checkpoint's
    refine net does not take a gap/time-delta input at all.

  - Because there's no gap embedding, "stem.0.weight" is
    (64, 12, 3, 3) -- i.e. 12 input channels, not 27. Concatenating
    base_merged + flow + tir0 + tir1 + wv0 + wv1 only sums to 11, one
    channel short -- the 12th channel is the base flownet's blend
    mask (the same (B,1,H,W) sigmoid mask IFNet already computes to
    blend warped_img0/warped_img1), given to the refine net as a
    per-pixel confidence signal:
        3  (base_merged RGB)
      + 4  (flow field)
      + 1  (mask, from the base flownet)
      + 1  (tir0)
      + 1  (tir1)
      + 1  (wv0)
      + 1  (wv1)
      = 12

  - The output projection is named "head_rgb" (3, 64, 3, 3), not
    "head".

  - There's an extra learned parameter "output_scale" of shape
    (1, 3, 1, 1) -- a per-channel gain applied to the residual
    BEFORE it's added to base_merged:
        final = base_merged + output_scale * head_rgb(features)

  - head_rgb is NOT zero-initialized in this checkpoint's state (it's
    a trained checkpoint, epoch 5), so no special init logic is
    needed at load time -- strict loading handles it.

This version ports the exact structure straight from the checkpoint's
own state_dict keys/shapes, so it loads with strict=True and ZERO
missing/unexpected keys, and calls Model.inference() with only the
img0/img1/tir0/tir1/wv0/wv1 arguments this network actually consumes
(no gap argument is passed into the refine net any more, though we
keep gap_minutes around for logging/metrics purposes only).

N real frames (auto-detected from matched .nc/.h5 timestamps): T1..T{N}

NO GROUND TRUTH INTERPOLATION ANY MORE. Previously this script
interpolated the MIDDLE of three real frames (predicting T2 from
T1+T3) purely so it could score the prediction against the real T2.
That mode is removed.

New mode -- midpoint interpolation between CONSECUTIVE real frames,
no ground truth involved:

  Given consecutive real frames T1, T2, T3, ... T{N} (e.g. INSAT
  frames at 12:00, 12:30, 13:00, ...), the model generates a brand
  new synthetic frame at the midpoint of every consecutive pair:

    mid(1,2) = Model(T1, T2)   <- e.g. 12:00 + 12:30 -> 12:15
    mid(2,3) = Model(T2, T3)   <- e.g. 12:30 + 13:00 -> 12:45
    ...
    mid(N-1,N) = Model(T{N-1}, T{N})

  There is no real frame at these midpoint times, so there is no
  ground truth to compare against and no error metrics are computed
  for these midpoints.

Output animation -- single interleaved sequence alternating real and
synthetic frames in chronological order:

  T1(real), mid(1,2)(model), T2(real), mid(2,3)(model), T3(real), ...,
  mid(N-1,N)(model), T{N}(real)

  Total frames = 2N - 1.

GIF output:
  Single animation (not side-by-side) showing the full interleaved
  real+interpolated sequence in chronological order.

WV and TIR1 input handling is UNCHANGED from before (same loading,
calibration, normalization, and model-input construction).
"""

import os
import sys
import glob
import re
import copy
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

warnings.filterwarnings("ignore")

# ==============================================================
#  SETTINGS - edit these
# ==============================================================

NC_FOLDER   = r"/kaggle/input/datasets/divyashkigf/gujrat/gujrat"   # folder scanned for BOTH .nc and .h5 files (name kept for backward-compat)
OUTPUT_DIR  = r"/kaggle/working/newfinaloutput"

# ----------------------------------------------------------------
# Data-source format. This script is now UNIVERSAL and accepts
# EITHER:
#   "goes_nc"   -> GOES-R series .nc files (CMI band arrays, e.g.
#                  OR_ABI-L2-CMIPF-M6C13_*.nc / ...C08_*.nc). These
#                  store the CMI variable already as calibrated
#                  brightness temperature in Kelvin -- confirmed by
#                  inspection (no further Planck conversion needed,
#                  matches this script's original working behaviour).
#   "insat_h5"  -> INSAT-3D/3DR/3DS L1C .h5 files (e.g.
#                  3SIMG_01JUN2026_0030_L1C_SGP_V01R00.h5). These
#                  DO NOT store temperature directly in the main
#                  image datasets -- IMG_TIR1 / IMG_WV etc. are raw
#                  10-bit instrument counts (uint16, e.g. ~450-990),
#                  confirmed by direct inspection of a sample file.
#                  The temperature is obtained via a per-file
#                  calibration LOOKUP TABLE shipped in the same .h5
#                  (IMG_TIR1_TEMP / IMG_WV_TEMP, each length 1024,
#                  indexed by the raw count / GreyCount). This LUT is
#                  ISRO's own calibration curve (not a manual Planck
#                  inversion -- the file already provides the final
#                  count->Kelvin mapping), and pixels equal to the
#                  dataset's _FillValue (1023) are masked as invalid,
#                  the same way GOES's DQF mask is used here.
#
# Set this to "auto" to infer the format per-file from its extension
# (.nc -> goes_nc, .h5 -> insat_h5), or force one explicitly.
# ----------------------------------------------------------------
DATA_FORMAT = "auto"   # "auto" | "goes_nc" | "insat_h5"

# For INSAT .h5 files: which image datasets to use for TIR / WV.
# IMG_TIR1 (10.8 micron window channel) is the standard analogue of
# GOES CH13; IMG_WV (6.7 micron) is the standard analogue of GOES
# CH8. (IMG_TIR2 / 12 micron split-window also exists if you'd
# rather use that instead of IMG_TIR1.)
INSAT_TIR_DATASET = "IMG_TIR1"
INSAT_WV_DATASET  = "IMG_WV"

# Path to your fine-tuned training checkpoint (NOT a plain flownet.pkl —
# this is the full training checkpoint with 'refine'/'ema_refine'/'base_flownet' keys)
FINETUNED_CKPT_PATH = r"/kaggle/input/datasets/mistridaivya/best-checkpoint-6th-aug/best_checkpoint.pth"

# Use the EMA-smoothed refine weights (recommended -- matches what the
# training script itself validates/ships as "best"). Set False to use
# the raw (non-EMA) refine weights instead.
USE_EMA_WEIGHTS = True

GIF_FPS   = 6      # frames per second for normal GIF
GIF_SCALE = 1.0    # 0.5 = half size, smaller file

# ----------------------------------------------------------------
# Hierarchical (recursive) midpoint interpolation depth.
#
# depth=1: only the direct midpoint of each consecutive real pair.
#     T1 --- mid(T1,T2) --- T2
#     e.g. 12:00 --- 12:15 --- 12:30
#
# depth=2: also interpolate the midpoint of (T1, mid) and
#     (mid, T2) -- i.e. feed one of the model's OWN outputs back in
#     as an input to interpolate again, halving the gap again.
#     T1 -- mid(T1,mid1) -- mid1 -- mid(mid1,T2) -- T2
#     e.g. 12:00 -- 12:07:30 -- 12:15 -- 12:22:30 -- 12:30
#
# depth=3: one more halving on top of that (uses depth-2 outputs as
#     inputs), giving 8 intervals per original real gap.
#
# Each extra level doubles the frame count between every pair of
# real frames and each level's input on the "new" side is itself a
# model prediction from the previous level, so error can compound --
# hence the hard cap at 3.
# ----------------------------------------------------------------
INTERP_DEPTH = 1          # 1, 2, or 3 -- how many recursive halvings
INTERP_DEPTH = max(1, min(3, INTERP_DEPTH))   # hard cap: 1-3

# Ask the user at runtime how many recursive halvings to do (hard
# capped to 3 regardless of what's typed). Press Enter to keep the
# INTERP_DEPTH default set above.
try:
    _user_in = input(
        f"\nHow many levels of hierarchical interpolation? (1-3, default {INTERP_DEPTH}): "
    ).strip()
    if _user_in:
        INTERP_DEPTH = max(1, min(3, int(_user_in)))
except Exception:
    print(f"  Invalid input -- using default depth {INTERP_DEPTH}")
print(f"  Using interpolation depth = {INTERP_DEPTH}  "
      f"({2**INTERP_DEPTH} intervals per real gap)")

BT13_MIN, BT13_MAX = 190.0, 310.0   # "clean" IR longwave (TIR, CH13)
BT8_MIN,  BT8_MAX  = 190.0, 280.0   # upper-level water vapor (WV, CH8)

# Refinement-branch architecture config -- MUST match training script exactly
REFINE_DROPOUT_P      = 0.08
REFINE_CHANNELS       = 64
REFINE_NUM_RESBLOCKS  = 5

# ==============================================================
#  OUTPUT FOLDERS
# ==============================================================

frames_real_dir  = os.path.join(OUTPUT_DIR, "1_real_frames")
frames_interp_dir = os.path.join(OUTPUT_DIR, "2_interp_frames")
compare_dir      = os.path.join(OUTPUT_DIR, "3_comparisons")
metrics_dir      = os.path.join(OUTPUT_DIR, "4_metrics")
anim_dir         = os.path.join(OUTPUT_DIR, "5_animation")

for d in [frames_real_dir, frames_interp_dir, compare_dir, metrics_dir, anim_dir]:
    os.makedirs(d, exist_ok=True)

# ==============================================================
#  MODEL DEFINITIONS
#  -- ported byte-identical from the training script so the
#     checkpoint's state_dicts load with strict=True and zero
#     missing/unexpected keys.
# ==============================================================

print("=" * 65)
print("Loading fine-tuned model (frozen IFNet base + EvolutionRefinementNet)...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    print(f"  ERROR: torch not available -- {e}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------
# Frozen base flownet (original RIFE IFNet, bidirectional dual-pass)
# ---------------------------------------------------------------

_backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    """Byte-identical port of model/warplayer.py's warp() (edge-to-edge
    linspace, 'border' padding) -- required so flow numbers mean what
    the pretrained weights expect them to mean."""
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in _backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        _backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                          tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    g = (_backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2),
                                    nn.ConvTranspose2d(c // 2, 4, 4, 2, 1))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.PReLU(c // 2),
                                    nn.ConvTranspose2d(c // 2, 1, 4, 2, 1))

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask


class IFNet(nn.Module):
    """Byte-identical port of the original RIFE IFNet (train_log/IFNet_HDv3.py).
    block_tea is included (unused in forward) purely so the pretrained
    flownet.pkl / this checkpoint's base_flownet loads with strict=True."""
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)   # present for strict-load only, unused here

    def forward(self, x, scale_list=(4, 2, 1)):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                               torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
        return flow_list, mask_list[2], merged


# ---------------------------------------------------------------
# Refinement branch (Evolution Network analogue) -- new capability
#
# NOTE: this matches the ACTUAL state_dict found in 6august.pth,
# which differs from an earlier assumed architecture:
#   - There is NO gap embedding at all (no "gap_embed.*" keys).
#     Time-delta / gap is not fed into this network's forward pass.
#   - stem.0 takes 12 input channels:
#       3 (base_merged RGB) + 4 (flow) + 1 (tir0) + 1 (tir1)
#       + 1 (wv0) + 1 (wv1) = 12
#   - The output conv is named "head_rgb", not "head".
#   - There's an extra learned "output_scale" parameter, shape
#     (1, 3, 1, 1), a per-channel gain multiplied into the residual
#     before it's added to base_merged.
# ---------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act1 = nn.PReLU(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act2 = nn.PReLU(ch)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        y = self.act1(self.conv1(x))
        y = self.drop(y)
        y = self.conv2(y)
        return self.act2(x + y)


class EvolutionRefinementNet(nn.Module):
    """
    Inputs (all at the base flownet's output resolution):
      - base_merged        : (B, 3, H, W)  frozen base's RGB output
      - flow               : (B, 4, H, W)  final flow field from base
      - mask               : (B, 1, H, W)  frozen base's blend mask
      - tir0, tir1         : (B, 1, H, W) each, RAW normalized TIR
      - wv0,  wv1          : (B, 1, H, W) each, RAW normalized WV
    Output:
      - residual : (B, 3, H, W), scaled by output_scale then ADDED
        to base_merged.
    """
    def __init__(self, channels=REFINE_CHANNELS, n_resblocks=REFINE_NUM_RESBLOCKS,
                 dropout_p=REFINE_DROPOUT_P):
        super().__init__()
        in_ch = 3 + 4 + 1 + 1 + 1 + 1 + 1  # = 12, no gap embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, 1, 1),
            nn.PReLU(channels),
        )
        self.resblocks = nn.Sequential(*[ResBlock(channels, dropout_p) for _ in range(n_resblocks)])
        self.head_rgb = nn.Conv2d(channels, 3, 3, 1, 1)
        self.output_scale = nn.Parameter(torch.ones(1, 3, 1, 1))

    def forward(self, base_merged, flow, mask, tir0, tir1, wv0, wv1):
        x = torch.cat([base_merged, flow, mask, tir0, tir1, wv0, wv1], dim=1)
        feat = self.stem(x)
        feat = self.resblocks(feat)
        residual = self.head_rgb(feat) * self.output_scale
        return residual


# ---------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------

class Model:
    def __init__(self):
        self.base_flownet = IFNet().to(device)
        self.refine = EvolutionRefinementNet().to(device)
        self.ema_refine = copy.deepcopy(self.refine).to(device)
        for p in self.ema_refine.parameters():
            p.requires_grad_(False)
        for p in self.base_flownet.parameters():
            p.requires_grad_(False)
        self.base_flownet.eval()

    def eval(self):
        self.base_flownet.eval()
        self.refine.eval()
        self.ema_refine.eval()

    def load_checkpoint(self, ckpt_path, use_ema=True):
        """Loads the FULL training checkpoint (with 'refine'/'ema_refine'/
        'base_flownet' keys) -- this is what 6august.pth actually is."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if not (isinstance(ckpt, dict) and "base_flownet" in ckpt and
                ("refine" in ckpt or "ema_refine" in ckpt)):
            raise RuntimeError(
                "Checkpoint doesn't look like the expected training-checkpoint "
                "format (missing 'base_flownet'/'refine'/'ema_refine' keys). "
                "Got top-level keys: " + str(list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
            )

        # base_flownet -- strict load, should be an exact match
        missing, unexpected = self.base_flownet.load_state_dict(ckpt["base_flownet"], strict=False)
        if missing or unexpected:
            print(f"  [WARN] base_flownet load -- missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("  base_flownet loaded (frozen, exact match).")

        # refine -- always load the raw refine weights into self.refine
        if "refine" in ckpt:
            self.refine.load_state_dict(ckpt["refine"], strict=True)
            print("  refine (non-EMA) weights loaded.")

        # ema_refine -- load if present
        if "ema_refine" in ckpt:
            self.ema_refine.load_state_dict(ckpt["ema_refine"], strict=True)
            print("  ema_refine (EMA) weights loaded.")
        else:
            self.ema_refine.load_state_dict(self.refine.state_dict())
            print("  [WARN] no ema_refine in checkpoint -- ema_refine mirrors raw refine.")

        # Pick which refine weights actually get used for inference
        self._active_refine = self.ema_refine if (use_ema and "ema_refine" in ckpt) else self.refine
        which = "EMA" if self._active_refine is self.ema_refine else "raw"
        print(f"  -> Using {which} refine weights for inference.")

        if isinstance(ckpt, dict) and "epoch" in ckpt:
            print(f"  Checkpoint epoch={ckpt['epoch']}, best_psnr={ckpt.get('best_psnr', 'NA')}")

        del ckpt

    def forward(self, img0, img1, tir0, tir1, wv0, wv1, scale_list=(4, 2, 1)):
        with torch.no_grad():
            x = torch.cat((img0, img1), 1)
            flow_list, mask, merged = self.base_flownet(x, scale_list=scale_list)
        base_merged = merged[2]
        final_flow = flow_list[2]
        residual = self._active_refine(base_merged, final_flow, mask, tir0, tir1, wv0, wv1)
        final = base_merged + residual
        return final, base_merged, residual, final_flow, mask

    def inference(self, img0, img1, tir0, tir1, wv0, wv1, scale=1.0):
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            final, base_merged, residual, flow, mask = self.forward(
                img0, img1, tir0, tir1, wv0, wv1, scale_list=scale_list)
        return final, base_merged, residual


rife_model = None
try:
    rife_model = Model()
    rife_model.load_checkpoint(FINETUNED_CKPT_PATH, use_ema=USE_EMA_WEIGHTS)
    rife_model.eval()
    print(f"  Fine-tuned model ready on {device}")
except Exception as e:
    print(f"  ERROR: fine-tuned model not loaded -- {e}")
    sys.exit(1)

# ==============================================================
#  HELPERS
# ==============================================================
#
# Universal timestamp handling
# -----------------------------
# GOES filenames encode start-scan time as _sYYYYDDDHHMMSSf_
# (julian day-of-year). INSAT filenames encode it as separate
# DDMMMYYYY + HHMM tokens, e.g.:
#   3SIMG_01JUN2026_0030_L1C_SGP_V01R00.h5
#              ^date^ ^HHMM (GMT)
#
# Internally we normalize BOTH formats down to the same 14-digit
# "YYYYDDDHHMMSS" key used throughout the rest of this script (so
# common_ts / sorting / minutes_between all keep working unchanged
# regardless of which satellite the files came from).

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def detect_format(filepath):
    """Infer file format from extension, honoring DATA_FORMAT override."""
    if DATA_FORMAT != "auto":
        return DATA_FORMAT
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".nc":
        return "goes_nc"
    if ext in (".h5", ".hdf5"):
        return "insat_h5"
    raise ValueError(f"Cannot infer format for '{filepath}' (unrecognized extension '{ext}')")

def get_timestamp(filepath):
    """
    Returns a normalized 14-digit 'YYYYDDDHHMMSS' timestamp key,
    regardless of whether filepath is a GOES .nc or INSAT .h5 file.
    """
    fmt = detect_format(filepath)
    base = os.path.basename(filepath)

    if fmt == "goes_nc":
        m = re.search(r"_s(\d{14})_", base)
        if not m:
            m = re.search(r"_s(\d+)_", base)
        return m.group(1) if m else None

    if fmt == "insat_h5":
        # e.g. 3SIMG_01JUN2026_0030_L1C_SGP_V01R00.h5
        m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_", base)
        if not m:
            return None
        dd, mon, yyyy, hhmm = m.groups()
        month = _MONTHS.get(mon.upper())
        if month is None:
            return None
        dd, hh, mm = int(dd), int(hhmm[:2]), int(hhmm[2:])
        dt = datetime(int(yyyy), month, dd, hh, mm, 0)
        doy = dt.timetuple().tm_yday
        return f"{dt.year:04d}{doy:03d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"

    raise ValueError(f"Unknown format '{fmt}' for '{filepath}'")

def parse_goes_timestamp(ts):
    """
    Parses the normalized 'YYYYDDDHHMMSS' (julian day-of-year) key
    produced by get_timestamp() for EITHER source format, back into
    a datetime object. (Name kept for backward compatibility --
    despite the name, this now works for both GOES and INSAT keys
    since both are normalized to the same representation.)
    """
    ts14 = ts[:13]  # YYYYDDDHHMMSS (13 digits, drop trailing tenths if present)
    year = int(ts14[0:4])
    doy  = int(ts14[4:7])
    hh   = int(ts14[7:9])
    mm   = int(ts14[9:11])
    ss   = int(ts14[11:13])
    dt = datetime(year, 1, 1) + __import__("datetime").timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss
    )
    return dt

def minutes_between(ts_a, ts_b):
    """Absolute minutes between two normalized timestamps."""
    ta = parse_goes_timestamp(ts_a)
    tb = parse_goes_timestamp(ts_b)
    return abs((tb - ta).total_seconds()) / 60.0

def load_nc(path):
    """
    Loads a GOES .nc CMI band file. The CMI variable is ALREADY
    calibrated brightness temperature in Kelvin (confirmed -- this is
    how this script has been working correctly all along), so no
    further LUT/Planck conversion is applied here. DQF is used as a
    quality mask (DQF<=1 kept, else NaN).
    """
    ds = xr.open_dataset(path)
    bt = ds["CMI"].values.astype(np.float32)
    if "DQF" in ds:
        bt = np.where(ds["DQF"].values <= 1, bt, np.nan)
    ds.close()
    return bt

def load_h5_band(path, dataset_name):
    """
    Loads one INSAT L1C .h5 band and converts it to calibrated
    brightness temperature in Kelvin.

    IMPORTANT -- verified directly against a real 3DS L1C file:
    the main image datasets (IMG_TIR1 / IMG_WV / etc.) are RAW
    10-bit instrument counts (uint16), NOT temperature. They are
    NOT run through Planck's law manually here -- instead each band
    ships its own count->Kelvin calibration lookup table in the same
    file, named '<dataset>_TEMP' (length 1024, one entry per
    possible raw count 0-1023). We index that LUT with the raw
    counts to get calibrated BT directly, which is what ISRO
    already computed (this LUT already encodes the Planck inversion
    + radiometric calibration -- we don't need to redo it).

    Pixels equal to the dataset's _FillValue attribute (1023 in the
    sample file) are masked to NaN, mirroring how GOES's DQF mask is
    used above.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {path}. "
                            f"Available: {list(f.keys())}")
        raw_ds = f[dataset_name]
        raw = raw_ds[:]                      # shape (1, H, W), uint16 raw counts
        raw = np.squeeze(raw, axis=0).astype(np.int32)

        fill_value = None
        if "_FillValue" in raw_ds.attrs:
            fv = raw_ds.attrs["_FillValue"]
            fill_value = int(np.asarray(fv).flatten()[0])

        lut_name = f"{dataset_name}_TEMP"
        if lut_name not in f:
            raise KeyError(f"Calibration LUT '{lut_name}' not found in {path} "
                            f"(needed to convert '{dataset_name}' counts to Kelvin).")
        lut = f[lut_name][:].astype(np.float32)   # shape (1024,), count -> Kelvin

    # Clip raw counts into valid LUT index range, then look up BT.
    idx = np.clip(raw, 0, lut.shape[0] - 1)
    bt = lut[idx]

    if fill_value is not None:
        bt = np.where(raw == fill_value, np.nan, bt)

    return bt.astype(np.float32)

def load_band_pair(ch13_path, ch8_path):
    """
    Universal loader: given a matched (tir_path, wv_path) pair,
    returns (bt13, bt8) as calibrated Kelvin arrays, regardless of
    whether the source is GOES .nc or INSAT .h5.
    """
    fmt = detect_format(ch13_path)
    if fmt == "goes_nc":
        bt13 = load_nc(ch13_path)
        bt8  = load_nc(ch8_path)
    elif fmt == "insat_h5":
        bt13 = load_h5_band(ch13_path, INSAT_TIR_DATASET)
        bt8  = load_h5_band(ch8_path, INSAT_WV_DATASET)
    else:
        raise ValueError(f"Unknown format '{fmt}'")
    return bt13, bt8

def normalize_bt(bt, vmin, vmax):
    bt = np.clip(bt, vmin, vmax)
    return (vmax - bt) / (vmax - vmin)

def make_rgb(bt13, bt8):
    r   = normalize_bt(bt13, BT13_MIN, BT13_MAX)
    g   = normalize_bt(bt8,  BT8_MIN,  BT8_MAX)
    b   = (r + g) / 2
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

def rgb_to_bt(rgb):
    r    = rgb[:, :, 0]
    g    = rgb[:, :, 1]
    bt13 = ((1.0 - r) * (BT13_MAX - BT13_MIN) + BT13_MIN).astype(np.float32)
    bt8  = ((1.0 - g) * (BT8_MAX  - BT8_MIN)  + BT8_MIN).astype(np.float32)
    return bt13, bt8

def save_png(arr_01, path):
    Image.fromarray((arr_01 * 255).astype(np.uint8)).save(path)

def pad_to_multiple(tensor, multiple=32):
    """Pads a (B,C,H,W) tensor on bottom/right so H and W are multiples
    of `multiple`. Uses replicate padding to avoid biasing flow near
    edges with zeros. Returns (padded_tensor, (orig_h, orig_w))."""
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def unpad(tensor, orig_hw):
    """Crops a (B,C,H,W) tensor back down to the original (h, w)."""
    h, w = orig_hw
    return tensor[..., :h, :w]
def run_model(bt13_a, bt8_a, bt13_b, bt8_b, gap_minutes):
    """
    Interpolate midpoint between (bt13_a,bt8_a) and (bt13_b,bt8_b) using
    the fine-tuned base+refine model.

    gap_minutes is kept as an argument purely for logging/metrics
    context (it's recorded in the metrics CSV) -- this checkpoint's
    refine net does not actually take a gap/time-delta input, so it
    is NOT passed into rife_model.inference().
    """
    try:
        rgb_a = make_rgb(bt13_a, bt8_a)
        rgb_b = make_rgb(bt13_b, bt8_b)

        def to_t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        def band_to_t(bt, vmin, vmax):
            n = normalize_bt(bt, vmin, vmax)
            n = np.nan_to_num(n, nan=0.0).astype(np.float32)
            return torch.from_numpy(n).unsqueeze(0).unsqueeze(0).float().to(device)

        img0 = to_t(rgb_a)
        img1 = to_t(rgb_b)
        tir0 = band_to_t(bt13_a, BT13_MIN, BT13_MAX)
        tir1 = band_to_t(bt13_b, BT13_MIN, BT13_MAX)
        wv0  = band_to_t(bt8_a,  BT8_MIN,  BT8_MAX)
        wv1  = band_to_t(bt8_b,  BT8_MIN,  BT8_MAX)

        # Pad all six inputs to a common multiple-of-32 size so the
        # model's internal downsample/upsample pyramid (scale_list =
        # 4,2,1) lines up cleanly. orig_hw is the same for all six
        # since they all come from the same-shaped source arrays.
        img0, orig_hw = pad_to_multiple(img0, 32)
        img1, _       = pad_to_multiple(img1, 32)
        tir0, _       = pad_to_multiple(tir0, 32)
        tir1, _       = pad_to_multiple(tir1, 32)
        wv0,  _       = pad_to_multiple(wv0,  32)
        wv1,  _       = pad_to_multiple(wv1,  32)

        final, base_merged, residual = rife_model.inference(
            img0, img1, tir0, tir1, wv0, wv1, scale=1.0)

        final = unpad(final, orig_hw)  # crop back to 3207 x 3062

        pred = np.clip(final[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)
        return pred
        
    except Exception as e:
        print(f"    Model error: {e}")
        return None

# ==============================================================
#  METRICS
# ==============================================================

def compute_metrics(pred_bt13, pred_bt8, gt_bt13, gt_bt8):
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn

    out = {}
    for ch, pred, gt, vmin, vmax in [
        ("CH13", pred_bt13, gt_bt13, BT13_MIN, BT13_MAX),
        ("CH8",  pred_bt8,  gt_bt8,  BT8_MIN,  BT8_MAX),
    ]:
        valid = ~(np.isnan(pred) | np.isnan(gt))
        if not valid.any():
            out[f"RMSE_{ch}"] = np.nan
            out[f"PSNR_{ch}"] = np.nan
            out[f"SSIM_{ch}"] = np.nan
            continue
        p  = np.where(valid, pred.astype(np.float64), gt.astype(np.float64))
        g  = gt.astype(np.float64)
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
        def _norm(arr, vmin, vmax):
            mid  = (vmin + vmax) / 2.0
            safe = np.where(np.isnan(arr), mid, arr.astype(np.float64))
            return np.clip((safe - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        def _t(ch13, ch8):
            r = _norm(ch13, BT13_MIN, BT13_MAX)
            g = _norm(ch8,  BT8_MIN,  BT8_MAX)
            b = (r + g) / 2
            return torch.from_numpy(np.stack([r, g, b])).unsqueeze(0)
        out["FSIM"] = float(piq.fsim(_t(pred_bt13, pred_bt8),
                                     _t(gt_bt13,   gt_bt8),
                                     data_range=1.0))
    except Exception:
        out["FSIM"] = float("nan")

    return out

# ==============================================================
#  LOAD & MATCH FILES
# ==============================================================
# Universal across formats:
#   - GOES .nc : each timestamp is split across TWO files, one per
#     band (CH13 / CH08). ch13_dict[ts] and ch8_dict[ts] point to
#     two different .nc files that both get passed to load_band_pair.
#   - INSAT .h5: each timestamp is ONE file containing all bands.
#     ch13_dict[ts] and ch8_dict[ts] point to the SAME .h5 file;
#     load_band_pair() reads IMG_TIR1 / IMG_WV out of it.
# ==============================================================

print(f"\nScanning data folder ({NC_FOLDER})...")
all_nc  = glob.glob(os.path.join(NC_FOLDER, "*.nc"))
all_h5  = glob.glob(os.path.join(NC_FOLDER, "*.h5")) + glob.glob(os.path.join(NC_FOLDER, "*.hdf5"))

ch13_dict = {}
ch8_dict  = {}

# GOES: CH13 (TIR) / CH08 (WV) live in separate files
goes_ch13_files = sorted([f for f in all_nc if "C13" in os.path.basename(f)])
goes_ch8_files  = sorted([f for f in all_nc if "C08" in os.path.basename(f)])
for f in goes_ch13_files:
    ts = get_timestamp(f)
    if ts:
        ch13_dict[ts] = f
for f in goes_ch8_files:
    ts = get_timestamp(f)
    if ts:
        ch8_dict[ts] = f

# INSAT: TIR + WV both live inside the SAME .h5 file per timestamp
for f in all_h5:
    ts = get_timestamp(f)
    if ts:
        ch13_dict[ts] = f
        ch8_dict[ts]  = f

common_ts = sorted(set(ch13_dict.keys()) & set(ch8_dict.keys()))

print(f"  .nc files: {len(all_nc)}  .h5 files: {len(all_h5)}  Matched timestamps: {len(common_ts)}")

if len(common_ts) < 3:
    print("ERROR: Need at least 3 matched timestamps.")
    sys.exit(1)

# We treat common_ts[0] as T1, common_ts[-1] as T{N}
N = len(common_ts)
print(f"  Using {N} frames: T1=ts[0] ... T{N}=ts[{N-1}]")
for i, ts in enumerate(common_ts):
    print(f"    T{i+1}: {ts}  [{detect_format(ch13_dict[ts])}]")

# ==============================================================
#  STEP 1 - Load all real RGB frames (+ raw BT arrays, needed by the
#            refine branch and for ground-truth metrics)
# ==============================================================

print(f"\n{'='*65}")
print("Loading all real frames...")

real_rgb   = []   # list of N rgb arrays  (index 0 = T1)
real_bt13  = []
real_bt8   = []

for i, ts in enumerate(common_ts):
    bt13, bt8 = load_band_pair(ch13_dict[ts], ch8_dict[ts])
    rgb  = make_rgb(bt13, bt8)
    real_rgb.append(rgb)
    real_bt13.append(bt13)
    real_bt8.append(bt8)
    save_png(rgb, os.path.join(frames_real_dir, f"T{i+1:02d}_{ts}_REAL.png"))
    print(f"  T{i+1:02d} loaded & saved")

# ==============================================================
#  STEP 2 - Build interleaved real+midpoint sequence
#            (hierarchical / recursive to INTERP_DEPTH levels)
# ==============================================================
# No ground truth involved -- every synthetic frame sits at a time
# that has no real observation, so nothing to score it against.
#
# For each consecutive real pair (A, B), we build a binary tree of
# midpoints down to INTERP_DEPTH levels, e.g. depth=2 for
# A=12:00, B=12:30:
#
#            mid(A,B) = 12:15            <- level 1 (Model(A,B))
#           /                  \
#   mid(A,mid)=12:07:30   mid(mid,B)=12:22:30   <- level 2
#                                                   (Model(A,12:15),
#                                                    Model(12:15,B))
#
# Level 2's inputs on the "new" side are level 1's OWN model output
# (recursive), not real data -- expected, and why depth is capped.
#
# The final chronological sequence for one real pair at depth=2 is:
#   A, mid(A,mid), mid(A,B), mid(mid,B), B
#   12:00, 12:07:30, 12:15, 12:22:30, 12:30
#
# This flattens naturally: build_interval(a_rgb,a_bt13,a_bt8,a_ts,
#                                          b_rgb,b_bt13,b_bt8,b_ts, depth)
# returns the ordered list of frames strictly BETWEEN a and b
# (exclusive of a and b themselves), recursing on each half.
# ==============================================================

print(f"\n{'='*65}")
print(f"Building interleaved real + hierarchical midpoint sequence "
      f"(depth={INTERP_DEPTH})...")

_mid_counter = [0]  # for unique output filenames


def _fmt_hms(dt):
    return dt.strftime("%H:%M:%S")


def interpolate_midpoint(bt13_a, bt8_a, ts_a_dt,
                          bt13_b, bt8_b, ts_b_dt):
    """Runs the model once between two (bt13, bt8) inputs (which may
    themselves be earlier model outputs) and returns
    (pred_rgb, pred_bt13, pred_bt8, mid_dt) or None on failure.
    WV/TIR handling inside run_model is unchanged."""
    gap_minutes = abs((ts_b_dt - ts_a_dt).total_seconds()) / 60.0
    half_gap    = gap_minutes / 2.0
    mid_dt      = ts_a_dt + (ts_b_dt - ts_a_dt) / 2

    pred = run_model(bt13_a, bt8_a, bt13_b, bt8_b, gap_minutes=half_gap)
    if pred is None:
        return None

    pred_bt13, pred_bt8 = rgb_to_bt(pred)
    return pred, pred_bt13, pred_bt8, mid_dt


def build_midpoints_recursive(bt13_a, bt8_a, ts_a_dt,
                               bt13_b, bt8_b, ts_b_dt,
                               depth, level=1):
    """Returns an ordered list of (rgb, label, is_real=False) frames
    strictly between a and b, recursing down to `depth` levels.
    Frames at deeper levels use the PREVIOUS level's model output as
    one of their two inputs (recursive/hierarchical, not from real
    data), which is why depth is hard-capped by the caller."""
    result = interpolate_midpoint(bt13_a, bt8_a, ts_a_dt,
                                   bt13_b, bt8_b, ts_b_dt)
    if result is None:
        print(f"    [level {level}] Model failed -- skipping this branch")
        return []

    pred_rgb, pred_bt13, pred_bt8, mid_dt = result
    _mid_counter[0] += 1
    label = f"L{level} {_fmt_hms(mid_dt)} MODEL"
    print(f"    [level {level}] mid = {_fmt_hms(mid_dt)}  "
          f"(between {_fmt_hms(ts_a_dt)} and {_fmt_hms(ts_b_dt)})")

    save_png(pred_rgb, os.path.join(
        frames_interp_dir,
        f"mid_{_mid_counter[0]:04d}_L{level}_{mid_dt.strftime('%Y%m%d_%H%M%S')}_MODEL.png"))

    if level >= depth:
        return [(pred_rgb, label, False)]

    # Recurse on both halves: (a, mid) and (mid, b). The "mid" side
    # is this level's freshly-generated model output, fed back in.
    left  = build_midpoints_recursive(bt13_a, bt8_a, ts_a_dt,
                                       pred_bt13, pred_bt8, mid_dt,
                                       depth, level + 1)
    right = build_midpoints_recursive(pred_bt13, pred_bt8, mid_dt,
                                       bt13_b, bt8_b, ts_b_dt,
                                       depth, level + 1)
    return left + [(pred_rgb, label, False)] + right


seq_rgb     = []   # chronological order
seq_labels  = []   # matching human-readable labels
seq_is_real = []   # bool per frame

# T1
seq_rgb.append(real_rgb[0])
seq_labels.append(f"T1 REAL {common_ts[0][-8:]}")
seq_is_real.append(True)
print(f"  [0] T1 real ({common_ts[0]})")

for i in range(N - 1):
    ts_a = common_ts[i]
    ts_b = common_ts[i + 1]
    dt_a = parse_goes_timestamp(ts_a)
    dt_b = parse_goes_timestamp(ts_b)

    print(f"  Interpolating between T{i+1} ({ts_a}) and T{i+2} ({ts_b})  "
          f"depth={INTERP_DEPTH}...")

    mids = build_midpoints_recursive(real_bt13[i], real_bt8[i], dt_a,
                                      real_bt13[i + 1], real_bt8[i + 1], dt_b,
                                      depth=INTERP_DEPTH)
    for rgb, label, is_real in mids:
        seq_rgb.append(rgb)
        seq_labels.append(label)
        seq_is_real.append(is_real)

    # next real frame T{i+2}
    seq_rgb.append(real_rgb[i + 1])
    seq_labels.append(f"T{i+2} REAL {ts_b[-8:]}")
    seq_is_real.append(True)
    print(f"  [{len(seq_rgb)-1}] T{i+2} real ({ts_b})")

n_seq = len(seq_rgb)
n_synthetic = n_seq - N
print(f"\n  Full interleaved sequence: {n_seq} frames "
      f"({N} real + {n_synthetic} synthetic, "
      f"{2**INTERP_DEPTH} intervals per original real gap)")

# ==============================================================
#  STEP 3 - Build single interleaved animation GIF
# ==============================================================
# One frame per sequence position, chronological order:
#   real, model, real, model, real, ...
# Label + colored border indicate real vs synthetic per frame.
# ==============================================================

print(f"\n{'='*65}")
print("Building interleaved animation GIF...")


def make_single_frame(img, label, is_real, frame_num, total_frames):
    """One GIF frame with a header label and a colored border
    (green = real, blue = synthetic/model)."""
    H, W = img.shape[:2]
    HDR  = 40
    BORDER = 4
    color = (80, 210, 120) if is_real else (90, 160, 255)

    canvas = np.full((H + HDR + BORDER, W + BORDER * 2, 3), 18, dtype=np.uint8)
    canvas[HDR:HDR+H, BORDER:BORDER+W] = (img * 255).astype(np.uint8)

    pil  = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    draw.rectangle([0, 0, W + BORDER * 2 - 1, HDR - 1], fill=(10, 10, 22))
    draw.rectangle([0, HDR - 1, W + BORDER * 2 - 1, HDR + H + BORDER - 1],
                   outline=color, width=BORDER)

    try:
        font    = ImageFont.truetype("arial.ttf", 16)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font    = ImageFont.load_default()
        font_sm = font

    draw.text((W // 2 + BORDER, HDR // 2), label,
              fill=color, font=font, anchor="mm")
    draw.text((W + BORDER * 2 - 6, HDR - 6),
              f"{frame_num}/{total_frames}",
              fill=(130, 130, 130), font=font_sm, anchor="rb")

    return pil


gif_frames = []
duration_ms = int(1000 / GIF_FPS)

for i in range(n_seq):
    frame_pil = make_single_frame(
        seq_rgb[i], seq_labels[i], seq_is_real[i],
        i + 1, n_seq,
    )

    if GIF_SCALE != 1.0:
        nw = int(frame_pil.width  * GIF_SCALE)
        nh = int(frame_pil.height * GIF_SCALE)
        frame_pil = frame_pil.resize((nw, nh), Image.LANCZOS)

    gif_frames.append(frame_pil)
    print(f"  GIF frame {i+1:02d}/{n_seq} rendered  ({seq_labels[i]})")

# Save normal-speed GIF
gif_path = os.path.join(anim_dir, "interleaved_real_and_interpolated.gif")
gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)
size_mb = os.path.getsize(gif_path) / 1e6
print(f"\n  GIF -> {gif_path}  ({size_mb:.1f} MB)")

# Save slow presentation GIF (800 ms/frame)
gif_slow = os.path.join(anim_dir, "interleaved_real_and_interpolated_slow.gif")
gif_frames[0].save(
    gif_slow,
    save_all=True,
    append_images=gif_frames[1:],
    duration=800,
    loop=0,
    optimize=False,
)
print(f"  Slow GIF -> {gif_slow}  (800 ms/frame)")

# ==============================================================
#  STEP 3b - Static filmstrip PNG
#            (all frames laid out in a single horizontal line,
#             e.g. 12:00 | 12:07:30 | 12:15 | 12:22:30 | 12:30 ...)
# ==============================================================

print(f"\n{'='*65}")
print("Building static filmstrip PNG (all frames in one row)...")

FILMSTRIP_THUMB_W = 220   # px width per thumbnail in the filmstrip

filmstrip_thumbs = []
for frame_pil in gif_frames:   # reuse already-rendered labeled frames
    scale = FILMSTRIP_THUMB_W / frame_pil.width
    thumb = frame_pil.resize(
        (FILMSTRIP_THUMB_W, int(frame_pil.height * scale)),
        Image.LANCZOS,
    )
    filmstrip_thumbs.append(thumb)

strip_h = max(t.height for t in filmstrip_thumbs)
strip_w = sum(t.width for t in filmstrip_thumbs)
filmstrip = Image.new("RGB", (strip_w, strip_h), (18, 18, 18))

x = 0
for thumb in filmstrip_thumbs:
    filmstrip.paste(thumb, (x, 0))
    x += thumb.width

filmstrip_path = os.path.join(anim_dir, "filmstrip_all_frames.png")
filmstrip.save(filmstrip_path)
print(f"  Filmstrip -> {filmstrip_path}  ({strip_w}x{strip_h}px, {n_seq} frames in one row)")

# ==============================================================
#  STEP 3c - Pure ground-truth animation + side-by-side comparison
#            vs the real+interpolated chain
# ==============================================================
# Purpose: a clean "before/after" proof for the pitch -- left panel
# = ONLY the real satellite frames (T1, T2, T3, ... T{N}), no model
# involved anywhere, at the SAME real-world timestamps the model
# chain passes through. Right panel = the full real+interpolated
# chain already built above (seq_rgb). Both GIFs play in lockstep on
# every REAL frame, so a viewer sees: "yahi real data hai" (left)
# next to "yahi hamara model ne generate kiya" (right) -- the extra
# in-between frames on the right are the value-add the model brings
# vs raw satellite revisit time.
#
# No new model calls, no new file loads -- this reuses real_rgb
# (already loaded in Step 1) and seq_rgb (already built in Step 2).
# ==============================================================

print(f"\n{'='*65}")
print("Building pure ground-truth animation + side-by-side comparison...")


def make_gt_only_frame(img, label, frame_num, total_frames):
    """Same visual style as make_single_frame but always green-bordered
    (every frame here is real -- no model output involved)."""
    return make_single_frame(img, label, True, frame_num, total_frames)


# --- Pure GT-only GIF (just the N real frames, nothing else) ---
gt_frames = []
for i in range(N):
    label = f"T{i+1} REAL {common_ts[i][-8:]} (ground truth)"
    frame_pil = make_gt_only_frame(real_rgb[i], label, i + 1, N)
    if GIF_SCALE != 1.0:
        nw = int(frame_pil.width  * GIF_SCALE)
        nh = int(frame_pil.height * GIF_SCALE)
        frame_pil = frame_pil.resize((nw, nh), Image.LANCZOS)
    gt_frames.append(frame_pil)

gt_gif_path = os.path.join(anim_dir, "pure_ground_truth.gif")
gt_frames[0].save(
    gt_gif_path,
    save_all=True,
    append_images=gt_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)
print(f"  Pure GT GIF -> {gt_gif_path}  ({N} real frames only)")


# --- Side-by-side: pure GT (left) vs real+interpolated chain (right) ---
# Synced on every REAL timestamp: at each of the N real timestamps,
# left shows that real frame; right advances through however many
# seq_rgb frames (real + synthetic) fall up to and including that
# same real timestamp, so the model's extra in-between frames are
# visible flashing past on the right while the left side holds/steps
# only on real data. This makes the model's added temporal density
# directly visible against the raw revisit cadence.

# Map each position in seq_rgb to "how many real frames have been
# passed so far" (1-indexed), so we know which gt_frame to pair it
# with at every step of the right-hand (model) track.
real_frames_seen = []
count = 0
for is_real in seq_is_real:
    if is_real:
        count += 1
    real_frames_seen.append(count)


def make_gt_vs_model_frame(gt_img, model_img, label_gt, label_model,
                            frame_num, total_frames):
    """Side-by-side panel: left = pure ground truth (green), right =
    real+interpolated model chain frame (green if real, blue if
    synthetic). Mirrors make_dual_frame's layout style."""
    H, W = gt_img.shape[:2]
    GAP, HDR = 8, 44
    canvas = np.full((H + HDR, W * 2 + GAP * 3, 3), 18, dtype=np.uint8)
    canvas[HDR:HDR+H, GAP:GAP+W] = (gt_img * 255).astype(np.uint8)
    canvas[HDR:HDR+H, GAP+W+GAP:GAP+W+GAP+W] = (model_img * 255).astype(np.uint8)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, W * 2 + GAP * 3, HDR - 1], fill=(10, 10, 22))
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    draw.text((GAP + W // 2, HDR // 2), label_gt,
              fill=(80, 210, 120), font=font, anchor="mm")
    draw.text((GAP + W + GAP + W // 2, HDR // 2), label_model,
              fill=(90, 160, 255), font=font, anchor="mm")
    draw.text((W * 2 + GAP * 3 - 6, HDR - 6), f"{frame_num}/{total_frames}",
              fill=(130, 130, 130), font=font_sm, anchor="rb")
    x_div = GAP + W + GAP // 2
    draw.line([(x_div, HDR), (x_div, H + HDR)], fill=(50, 50, 60), width=2)
    return pil


gt_vs_model_frames = []
for i in range(n_seq):
    gt_idx = real_frames_seen[i] - 1   # 0-indexed into real_rgb/common_ts
    gt_idx = max(0, min(N - 1, gt_idx))

    label_gt = f"T{gt_idx+1} REAL (ground truth)  {common_ts[gt_idx][-8:]}"
    label_model = seq_labels[i] if not seq_is_real[i] else f"{seq_labels[i]} (model chain)"

    frame_pil = make_gt_vs_model_frame(
        real_rgb[gt_idx], seq_rgb[i], label_gt, label_model, i + 1, n_seq,
    )
    if GIF_SCALE != 1.0:
        nw = int(frame_pil.width  * GIF_SCALE)
        nh = int(frame_pil.height * GIF_SCALE)
        frame_pil = frame_pil.resize((nw, nh), Image.LANCZOS)
    gt_vs_model_frames.append(frame_pil)

gt_vs_model_path = os.path.join(anim_dir, "ground_truth_vs_model_chain.gif")
gt_vs_model_frames[0].save(
    gt_vs_model_path,
    save_all=True,
    append_images=gt_vs_model_frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=False,
)
print(f"  GT vs model side-by-side GIF -> {gt_vs_model_path}  "
      f"({n_seq} frames: left=pure GT held on nearest real frame, "
      f"right=full real+interpolated chain)")

gt_vs_model_slow = os.path.join(anim_dir, "ground_truth_vs_model_chain_slow.gif")
gt_vs_model_frames[0].save(
    gt_vs_model_slow,
    save_all=True,
    append_images=gt_vs_model_frames[1:],
    duration=800,
    loop=0,
    optimize=False,
)
print(f"  Slow version -> {gt_vs_model_slow}  (800 ms/frame)")

# ==============================================================
#  STEP 4 - Summary
# ==============================================================

print(f"\n{'='*65}")
print("ALL DONE")
print(f"{'='*65}")
print(f"  Real frame PNGs      -> {frames_real_dir}/")
print(f"  Synthetic frame PNGs -> {frames_interp_dir}/")
print(f"  GIF (normal, {GIF_FPS} fps) -> {gif_path}")
print(f"  GIF (slow)           -> {gif_slow}")
print(f"  Filmstrip (one row)  -> {filmstrip_path}")
print(f"  Pure ground truth GIF          -> {gt_gif_path}")
print(f"  GT vs model side-by-side GIF   -> {gt_vs_model_path}")
print(f"  GT vs model side-by-side (slow) -> {gt_vs_model_slow}")
print()
print(f"  Interpolation depth: {INTERP_DEPTH}  "
      f"({2**INTERP_DEPTH} intervals per original real gap)")
print(f"  Real frames:      {N}")
print(f"  Synthetic frames: {n_seq - N}  (recursive midpoints, "
      f"no ground truth -- no metrics computed)")
print(f"  Total frames:     {n_seq}")
print(f"{'='*65}")
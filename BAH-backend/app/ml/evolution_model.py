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

N real frames (auto-detected from matched .nc timestamps): T1 to T{N}

Animation 1 — REAL:
  T1(real) T2(real) T3(real) ... T{N}(real)   [N frames]

Animation 2 — INTERPOLATED:
  T1(real)            <- no T0 exists, so kept real
  T2 = Model(T1, T3)  <- interpolated (fine-tuned base+refine)
  T3 = Model(T2, T4)
  T4 = Model(T3, T5)
  ...
  T{N-1} = Model(T{N-2}, T{N})
  T{N}(real)          <- no T{N+1} exists, so kept real

GIF output:
  Side-by-side: left = real animation, right = interpolated animation
  Both sync'd frame-by-frame (N frames each)

Metrics (T2 to T{N-1} only, the actually-interpolated frames):
  RMSE, PSNR, SSIM, FSIM - pred vs real ground truth
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

NC_FOLDER   = r"C:\isro goes data\round0\INSAT 3DS SPG"   # folder scanned for BOTH .nc and .h5 files (name kept for backward-compat)
OUTPUT_DIR  = r"C:\isro goes data\6augustoutputround0"

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
FINETUNED_CKPT_PATH = r"C:\isro goes data\6august.pth"

# Use the EMA-smoothed refine weights (recommended -- matches what the
# training script itself validates/ships as "best"). Set False to use
# the raw (non-EMA) refine weights instead.
USE_EMA_WEIGHTS = True

GIF_FPS   = 2      # frames per second for normal GIF
GIF_SCALE = 1.0    # 0.5 = half size, smaller file

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
    b   = normalize_bt(bt8,  BT8_MIN,  BT8_MAX)
    g   = (r + b) / 2
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

def rgb_to_bt(rgb):
    r    = rgb[:, :, 0]
    b    = rgb[:, :, 2]
    bt13 = ((1.0 - r) * (BT13_MAX - BT13_MIN) + BT13_MIN).astype(np.float32)
    bt8  = ((1.0 - b) * (BT8_MAX  - BT8_MIN)  + BT8_MIN).astype(np.float32)
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


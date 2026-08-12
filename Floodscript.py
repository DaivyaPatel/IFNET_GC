"""
INSAT Flood-Tracking Interpolation Test Script -- HIERARCHICAL VERSION
=========================================================================
Same base pipeline as insat_flood_interp_test.py (TIR1/WV interpolation
via fine-tuned IFNet + EvolutionRefinementNet, flood RGB composites),
PLUS:

  1. FEATURED_COMPOSITE is now locked to "day_water" -- the flood-extent
     comparison GIF always uses that composite when it's available,
     instead of picking whichever composite happens to have full
     coverage first.

  2. HIERARCHICAL (RECURSIVE) INTERPOLATION.
     Level 1 is exactly what the base script already did: for every
     real triple (T[i-1], T[i], T[i+1]) the model predicts the middle
     frame from its two real neighbors, and that prediction is scored
     against the real T[i] (RMSE/PSNR/SSIM/FSIM) since ground truth
     exists at level 1.

     From level 2 onward there is NO ground truth for the new
     in-between timestamps (a frame at t=0.5 between real T0 and
     level-1-interpolated T1 was never actually acquired by the
     satellite), so those levels are metrics-free by construction --
     the script does not compute or report RMSE/PSNR/SSIM/FSIM for
     them, since there is nothing to score against. It still runs the
     model forward using the level-(k-1) frame sequence as input:
     level k takes EVERY adjacent pair from level k-1's frame sequence
     and predicts the frame exactly halfway between them (gap = half
     of the level-(k-1) inter-frame spacing), then interleaves those
     new predictions with the level-(k-1) frames. Frame count doubles
     (minus one) each level: N frames -> 2N-1 -> 4N-3 -> 8N-7 ...

     HOW MANY LEVELS: set via HIERARCHY_LEVELS below, or (if
     PROMPT_FOR_HIERARCHY_LEVELS=True) you'll be asked at runtime for
     an integer (1 = base interpolation only, no recursion; 2 = one
     round of halving; 3 = two rounds; etc). Each extra level doubles
     the animation's frame count and roughly doubles model inference
     time and compounds interpolation error (level k's input is
     level (k-1)'s OUTPUT, not real data, past level 1) -- so frames
     get smoother in count but each level is working from a slightly
     softer/blurrier source than the last. That's expected and is
     flagged in the printed summary, not hidden.

  3. COMBINED ANIMATION -- ALL LEVELS IN ONE GIF. After every requested
     level (1..n_levels) is built, ONE GIF is rendered that plays level
     1's complete frame sequence, then level 2's complete (now-larger)
     sequence, then level 3's, and so on -- so you visibly watch the
     animation get smoother as the hierarchy deepens, rather than only
     ever seeing the deepest level in isolation. Every frame is labelled
     with its level number, its (real or virtual) timestamp, and
     REAL/interp so it's always clear which frames were ever seen by
     the satellite and which are synthetic. All levels are rendered with
     the SAME composite (FEATURED_COMPOSITE, falling back through the
     composite list if that composite isn't available for every level).

Everything else (channel auto-detection, IFNet/EvolutionRefinementNet
definitions, checkpoint loading, warp(), pad_to_multiple()) is ported
byte-identical from the working script so the checkpoint keeps loading
strict=True with zero missing/unexpected keys.
"""

import os
import sys
import glob
import re
import copy
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ==============================================================
#  SETTINGS - edit these
# ==============================================================

H5_FOLDER   = r"/kaggle/input/datasets/divyashkigf/clouflood/central_cloud_crop"
OUTPUT_DIR  = r"/kaggle/working/floooooodd"

FINETUNED_CKPT_PATH = r"/kaggle/input/datasets/mistridaivya/best-checkpoint-6th-aug/best_checkpoint.pth"

USE_EMA_WEIGHTS = True

GIF_FPS   = 2
GIF_SCALE = 1.0

# Calibration ranges for the TWO channels the model actually consumes.
BT13_MIN, BT13_MAX = 190.0, 310.0   # TIR1
BT8_MIN,  BT8_MAX  = 190.0, 280.0   # WV

# Refinement-branch architecture config -- MUST match training script.
REFINE_DROPOUT_P      = 0.08
REFINE_CHANNELS       = 64
REFINE_NUM_RESBLOCKS  = 5

# Which composite is used for the final smooth animation. Falls back
# through FLOOD_COMPOSITES (in list order) if this one isn't available
# for every frame in your crop (e.g. missing VIS/SWIR at night).
FEATURED_COMPOSITE = "day_water"

# --- Hierarchy control ------------------------------------------------
# How many interpolation levels to run. 1 = only the base T[i-1],T[i+1]
# -> T[i] interpolation (identical to the non-hierarchical script,
# scored against ground truth). Each level above 1 halves the time gap
# again with NO ground truth available for the new frames.
#   1 -> just base interpolation (has metrics)
#   2 -> base + one halving pass (adds midpoints between every adjacent
#        pair from level 1, no metrics for those)
#   3 -> base + two halving passes
#   ... etc
# If PROMPT_FOR_HIERARCHY_LEVELS is True, HIERARCHY_LEVELS below is
# just the default offered at the prompt; enter a number when asked
# (or press Enter to accept the default).
PROMPT_FOR_HIERARCHY_LEVELS = True
HIERARCHY_LEVELS = 2

# ==============================================================
#  INSAT CHANNEL NOMENCLATURE (for auto-detection)
# ==============================================================
INSAT_CHANNEL_PATTERNS = {
    "TIR1":  [r"IMG_TIR1\b", r"\bTIR1\b"],
    "TIR2":  [r"IMG_TIR2\b", r"\bTIR2\b"],
    "MIR":   [r"IMG_MIR\b",  r"\bMIR\b"],
    "WV":    [r"IMG_WV\b",   r"\bWV\b"],
    "SWIR":  [r"IMG_SWIR\b", r"\bSWIR\b"],
    "VIS":   [r"IMG_VIS\b",  r"\bVIS\b"],
}

EMISSIVE_CHANNELS = {"TIR1", "TIR2", "MIR", "WV"}
REFLECTIVE_CHANNELS = {"VIS", "SWIR"}

MODEL_TIR_CHANNEL = "TIR1"
MODEL_WV_CHANNEL  = "WV"

# ==============================================================
#  OUTPUT FOLDERS
# ==============================================================

frames_real_dir     = os.path.join(OUTPUT_DIR, "1_real_frames")
frames_interp_dir    = os.path.join(OUTPUT_DIR, "2_interp_frames")
compare_dir          = os.path.join(OUTPUT_DIR, "3_comparisons")
metrics_dir          = os.path.join(OUTPUT_DIR, "4_metrics")
anim_dir             = os.path.join(OUTPUT_DIR, "5_animation")
flood_composite_dir  = os.path.join(OUTPUT_DIR, "6_flood_composites")
hierarchy_dir        = os.path.join(OUTPUT_DIR, "7_hierarchy_levels")

for d in [frames_real_dir, frames_interp_dir, compare_dir, metrics_dir,
          anim_dir, flood_composite_dir, hierarchy_dir]:
    os.makedirs(d, exist_ok=True)

# ==============================================================
#  MODEL DEFINITIONS -- byte-identical to the working script
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

_backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
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
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if not (isinstance(ckpt, dict) and "base_flownet" in ckpt and
                ("refine" in ckpt or "ema_refine" in ckpt)):
            raise RuntimeError(
                "Checkpoint doesn't look like the expected training-checkpoint "
                "format (missing 'base_flownet'/'refine'/'ema_refine' keys). "
                "Got top-level keys: " + str(list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
            )

        missing, unexpected = self.base_flownet.load_state_dict(ckpt["base_flownet"], strict=False)
        if missing or unexpected:
            print(f"  [WARN] base_flownet load -- missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("  base_flownet loaded (frozen, exact match).")

        if "refine" in ckpt:
            self.refine.load_state_dict(ckpt["refine"], strict=True)
            print("  refine (non-EMA) weights loaded.")

        if "ema_refine" in ckpt:
            self.ema_refine.load_state_dict(ckpt["ema_refine"], strict=True)
            print("  ema_refine (EMA) weights loaded.")
        else:
            self.ema_refine.load_state_dict(self.refine.state_dict())
            print("  [WARN] no ema_refine in checkpoint -- ema_refine mirrors raw refine.")

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
#  TIMESTAMP HANDLING (INSAT filename format)
# ==============================================================

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def get_timestamp(filepath):
    base = os.path.basename(filepath)
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_", base)
    if m:
        dd, mon, yyyy, hhmm = m.groups()
        month = _MONTHS.get(mon.upper())
        if month is not None:
            dd, hh, mm = int(dd), int(hhmm[:2]), int(hhmm[2:])
            dt = datetime(int(yyyy), month, dd, hh, mm, 0)
            doy = dt.timetuple().tm_yday
            return f"{dt.year:04d}{doy:03d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"

    print(f"  [WARN] Could not parse INSAT timestamp from '{base}', "
          f"using file mtime instead (gap-minutes math will be approximate).")
    dt = datetime.fromtimestamp(os.path.getmtime(filepath))
    doy = dt.timetuple().tm_yday
    return f"{dt.year:04d}{doy:03d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"


def parse_ts(ts):
    ts14 = ts[:13]
    year = int(ts14[0:4]); doy = int(ts14[4:7])
    hh = int(ts14[7:9]); mm = int(ts14[9:11]); ss = int(ts14[11:13])
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
    return dt


def ts_from_dt(dt):
    doy = dt.timetuple().tm_yday
    return f"{dt.year:04d}{doy:03d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"


def minutes_between(ts_a, ts_b):
    return abs((parse_ts(ts_b) - parse_ts(ts_a)).total_seconds()) / 60.0


def midpoint_ts(ts_a, ts_b):
    """Virtual timestamp exactly halfway between two (real or virtual)
    timestamps -- used to label hierarchy levels >= 2, where this
    instant was never actually acquired."""
    dt_a, dt_b = parse_ts(ts_a), parse_ts(ts_b)
    mid = dt_a + (dt_b - dt_a) / 2
    return ts_from_dt(mid)

# ==============================================================
#  CHANNEL AUTO-DETECTION
# ==============================================================

def detect_channels_in_h5(path):
    import h5py
    found = {}
    with h5py.File(path, "r") as f:
        dataset_names = list(f.keys())
        for canon, patterns in INSAT_CHANNEL_PATTERNS.items():
            for pat in patterns:
                match = None
                for name in dataset_names:
                    if name.upper().endswith("_TEMP"):
                        continue
                    if re.search(pat, name, re.IGNORECASE):
                        match = name
                        break
                if match:
                    found[canon] = match
                    break
    return found


def load_h5_band(path, dataset_name, emissive):
    import h5py
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {path}. "
                            f"Available: {list(f.keys())}")
        raw_ds = f[dataset_name]
        raw = raw_ds[:]
        raw = np.squeeze(raw, axis=0).astype(np.int32) if raw.ndim == 3 else raw.astype(np.int32)

        fill_value = None
        if "_FillValue" in raw_ds.attrs:
            fv = raw_ds.attrs["_FillValue"]
            fill_value = int(np.asarray(fv).flatten()[0])

        if not emissive:
            out = raw.astype(np.float32)
            if fill_value is not None:
                out = np.where(raw == fill_value, np.nan, out)
            return out

        lut_name = f"{dataset_name}_TEMP"
        if lut_name not in f:
            raise KeyError(f"Calibration LUT '{lut_name}' not found in {path} "
                            f"(needed to convert '{dataset_name}' counts to Kelvin).")
        lut = f[lut_name][:].astype(np.float32)

    idx = np.clip(raw, 0, lut.shape[0] - 1)
    bt = lut[idx]
    if fill_value is not None:
        bt = np.where(raw == fill_value, np.nan, bt)
    return bt.astype(np.float32)


def load_all_channels(path):
    detected = detect_channels_in_h5(path)
    if not detected:
        raise RuntimeError(f"No recognizable INSAT channels found in {path}. "
                            f"Check INSAT_CHANNEL_PATTERNS against this file's dataset names.")
    data = {}
    for canon, dataset_name in detected.items():
        emissive = canon in EMISSIVE_CHANNELS
        data[canon] = load_h5_band(path, dataset_name, emissive=emissive)
        kind = "BT (K)" if emissive else "raw counts"
        print(f"    {canon:5s} <- '{dataset_name}'  [{kind}]")
    return data

# ==============================================================
#  NORMALIZATION / COMPOSITE HELPERS
# ==============================================================

def normalize_bt(bt, vmin, vmax):
    bt = np.clip(bt, vmin, vmax)
    return (vmax - bt) / (vmax - vmin)


def normalize_reflective(arr, pmin=1, pmax=99):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(valid, [pmin, pmax])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def make_model_rgb(bt13, bt8):
    r = normalize_bt(bt13, BT13_MIN, BT13_MAX)
    g = normalize_bt(bt8, BT8_MIN, BT8_MAX)
    b = (r + g) / 2
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)


def rgb_to_bt(rgb):
    r = rgb[:, :, 0]; g = rgb[:, :, 1]
    bt13 = ((1.0 - r) * (BT13_MAX - BT13_MIN) + BT13_MIN).astype(np.float32)
    bt8 = ((1.0 - g) * (BT8_MAX - BT8_MIN) + BT8_MIN).astype(np.float32)
    return bt13, bt8


def save_png(arr_01, path):
    Image.fromarray((np.clip(arr_01, 0, 1) * 255).astype(np.uint8)).save(path)


# ---- Flood composites -----------------------------------------------

def composite_day_water(chans):
    if "VIS" not in chans or "SWIR" not in chans:
        return None
    vis = normalize_reflective(chans["VIS"])
    swir = normalize_reflective(chans["SWIR"])
    b = 1.0 - swir
    rgb = np.dstack((vis, vis, b))
    return np.clip(rgb, 0, 1).astype(np.float32)


def composite_pseudo_ndwi(chans):
    if "VIS" not in chans or "SWIR" not in chans:
        return None
    vis = normalize_reflective(chans["VIS"])
    swir = normalize_reflective(chans["SWIR"])
    denom = (vis + swir)
    denom = np.where(denom == 0, 1e-6, denom)
    ndwi = (vis - swir) / denom
    ndwi01 = np.clip((ndwi + 1) / 2, 0, 1)
    cmap = plt.get_cmap("BrBG")
    rgb = cmap(ndwi01)[:, :, :3].astype(np.float32)
    return rgb


def composite_day_microphysics(chans):
    if not all(k in chans for k in ("VIS", "SWIR", "TIR1")):
        return None
    vis = normalize_reflective(chans["VIS"])
    swir = normalize_reflective(chans["SWIR"])
    tir1 = normalize_bt(chans["TIR1"], BT13_MIN, BT13_MAX)
    rgb = np.dstack((vis, swir, tir1))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)


def composite_night_rain_proxy(chans):
    if not all(k in chans for k in ("MIR", "TIR1", "TIR2")):
        return None
    btd = chans["MIR"] - chans["TIR1"]
    btd_n = normalize_reflective(btd, pmin=2, pmax=98)
    tir1_n = normalize_bt(chans["TIR1"], BT13_MIN, BT13_MAX)
    tir2_n = normalize_bt(chans["TIR2"], BT13_MIN, BT13_MAX)
    rgb = np.dstack((btd_n, tir1_n, tir2_n))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)


FLOOD_COMPOSITES = [
    ("day_water",         "Day Natural/Water (VIS-VIS-SWIRinv)",  composite_day_water),
    ("pseudo_ndwi",        "Pseudo-NDWI (VIS,SWIR)",               composite_pseudo_ndwi),
    ("day_microphysics",   "Day Microphysics (VIS-SWIR-TIR1inv)",  composite_day_microphysics),
    ("night_rain_proxy",   "Night Flood/Rain-Cloud proxy (BTD-TIR1-TIR2)", composite_night_rain_proxy),
]
FLOOD_COMPOSITE_FUNCS = {k: fn for k, _, fn in FLOOD_COMPOSITES}


def build_flood_composites(chans, out_prefix, ts_label, quiet=False):
    built = {}
    for key, desc, fn in FLOOD_COMPOSITES:
        rgb = fn(chans)
        if rgb is None:
            if not quiet:
                print(f"    [skip] {desc}: missing required channel(s)")
            continue
        built[key] = rgb
        save_png(rgb, os.path.join(flood_composite_dir, f"{out_prefix}_{key}_{ts_label}.png"))
        if not quiet:
            print(f"    [ok]   {desc} -> saved")
    return built

# ==============================================================
#  MODEL PADDING HELPERS
# ==============================================================

def pad_to_multiple(tensor, multiple=32):
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def unpad(tensor, orig_hw):
    h, w = orig_hw
    return tensor[..., :h, :w]


def run_model(bt13_a, bt8_a, bt13_b, bt8_b):
    """Interpolates the exact midpoint between two TIR1/WV frames.
    Works identically whether the two input frames are real or
    themselves previously-interpolated (hierarchy level >= 2) -- the
    model has no notion of "real" vs "synthetic" input, it just sees
    two BT fields and predicts the frame between them."""
    try:
        rgb_a = make_model_rgb(bt13_a, bt8_a)
        rgb_b = make_model_rgb(bt13_b, bt8_b)

        def to_t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        def band_to_t(bt, vmin, vmax):
            n = normalize_bt(bt, vmin, vmax)
            n = np.nan_to_num(n, nan=0.0).astype(np.float32)
            return torch.from_numpy(n).unsqueeze(0).unsqueeze(0).float().to(device)

        img0 = to_t(rgb_a); img1 = to_t(rgb_b)
        tir0 = band_to_t(bt13_a, BT13_MIN, BT13_MAX)
        tir1 = band_to_t(bt13_b, BT13_MIN, BT13_MAX)
        wv0 = band_to_t(bt8_a, BT8_MIN, BT8_MAX)
        wv1 = band_to_t(bt8_b, BT8_MIN, BT8_MAX)

        img0, orig_hw = pad_to_multiple(img0, 32)
        img1, _ = pad_to_multiple(img1, 32)
        tir0, _ = pad_to_multiple(tir0, 32)
        tir1, _ = pad_to_multiple(tir1, 32)
        wv0, _ = pad_to_multiple(wv0, 32)
        wv1, _ = pad_to_multiple(wv1, 32)

        final, base_merged, residual = rife_model.inference(
            img0, img1, tir0, tir1, wv0, wv1, scale=1.0)
        final = unpad(final, orig_hw)

        pred = np.clip(final[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)
        return pred
    except Exception as e:
        print(f"    Model error: {e}")
        return None

# ==============================================================
#  METRICS -- ONLY meaningful/used at hierarchy level 1, where a real
#  ground-truth frame exists for every interpolated one.
# ==============================================================

def compute_metrics(pred_bt13, pred_bt8, gt_bt13, gt_bt8):
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn

    out = {}
    for ch, pred, gt, vmin, vmax in [
        ("TIR1", pred_bt13, gt_bt13, BT13_MIN, BT13_MAX),
        ("WV", pred_bt8, gt_bt8, BT8_MIN, BT8_MAX),
    ]:
        valid = ~(np.isnan(pred) | np.isnan(gt))
        if not valid.any():
            out[f"RMSE_{ch}"] = np.nan; out[f"PSNR_{ch}"] = np.nan; out[f"SSIM_{ch}"] = np.nan
            continue
        p = np.where(valid, pred.astype(np.float64), gt.astype(np.float64))
        g = gt.astype(np.float64)
        dr = float(vmax - vmin)
        out[f"RMSE_{ch}"] = float(np.sqrt(np.mean((p - g) ** 2)))
        pn = np.clip((p - vmin) / dr, 0, 1)
        gn = np.clip((g - vmin) / dr, 0, 1)
        out[f"PSNR_{ch}"] = float(psnr_fn(gn, pn, data_range=1.0))
        out[f"SSIM_{ch}"] = float(ssim_fn(gn, pn, data_range=1.0))

    out["RMSE_avg"] = float(np.nanmean([out.get("RMSE_TIR1", np.nan), out.get("RMSE_WV", np.nan)]))
    out["PSNR_avg"] = float(np.nanmean([out.get("PSNR_TIR1", np.nan), out.get("PSNR_WV", np.nan)]))
    out["SSIM_avg"] = float(np.nanmean([out.get("SSIM_TIR1", np.nan), out.get("SSIM_WV", np.nan)]))

    try:
        import piq
        def _norm(arr, vmin, vmax):
            mid = (vmin + vmax) / 2.0
            safe = np.where(np.isnan(arr), mid, arr.astype(np.float64))
            return np.clip((safe - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
        def _t(ch13, ch8):
            r = _norm(ch13, BT13_MIN, BT13_MAX); g = _norm(ch8, BT8_MIN, BT8_MAX); b = (r + g) / 2
            return torch.from_numpy(np.stack([r, g, b])).unsqueeze(0)
        out["FSIM"] = float(piq.fsim(_t(pred_bt13, pred_bt8), _t(gt_bt13, gt_bt8), data_range=1.0))
    except Exception:
        out["FSIM"] = float("nan")

    return out

# ==============================================================
#  SCAN FOLDER, DETECT CHANNELS, LOAD EVERYTHING
# ==============================================================

print(f"\nScanning data folder ({H5_FOLDER})...")
all_h5 = sorted(glob.glob(os.path.join(H5_FOLDER, "*.h5")) +
                 glob.glob(os.path.join(H5_FOLDER, "*.hdf5")))

if not all_h5:
    print("ERROR: No .h5/.hdf5 files found in H5_FOLDER.")
    sys.exit(1)

print(f"  Found {len(all_h5)} .h5 file(s)")

file_ts = {}
for f in all_h5:
    ts = get_timestamp(f)
    if ts:
        file_ts[ts] = f

common_ts = sorted(file_ts.keys())
N = len(common_ts)
print(f"  Using {N} timestamp(s):")
for i, ts in enumerate(common_ts):
    print(f"    T{i+1}: {ts}  <- {os.path.basename(file_ts[ts])}")

if N < 1:
    print("ERROR: No usable timestamped files.")
    sys.exit(1)

print(f"\n{'='*65}")
print("Auto-detecting INSAT channels per file...")
all_channels_by_ts = {}
for i, ts in enumerate(common_ts):
    print(f"  T{i+1:02d} ({os.path.basename(file_ts[ts])}):")
    chans = load_all_channels(file_ts[ts])
    all_channels_by_ts[ts] = chans
    missing_for_model = [c for c in (MODEL_TIR_CHANNEL, MODEL_WV_CHANNEL) if c not in chans]
    if missing_for_model:
        print(f"    [WARN] Missing channel(s) required by the model: {missing_for_model} "
              f"-- interpolation will be skipped for this timestamp.")

if N < 3:
    print(f"\n[NOTE] Only {N} timestamp(s) available -- can't run the "
          f"real-vs-interpolated comparison (needs >=3). Flood composites "
          f"will still be built for whatever timestamps you gave; skipping "
          f"the interpolation/metrics/hierarchy/animation stages.")

# ==============================================================
#  STEP 1 - Build flood composites + model RGB for every real frame
# ==============================================================

print(f"\n{'='*65}")
print("Building flood composites for all real frames...")

real_model_rgb = []
real_bt13 = []
real_bt8 = []
real_flood_composites = []

for i, ts in enumerate(common_ts):
    chans = all_channels_by_ts[ts]

    if MODEL_TIR_CHANNEL in chans and MODEL_WV_CHANNEL in chans:
        bt13 = chans[MODEL_TIR_CHANNEL]
        bt8 = chans[MODEL_WV_CHANNEL]
    else:
        bt13 = bt8 = None

    real_bt13.append(bt13)
    real_bt8.append(bt8)

    if bt13 is not None and bt8 is not None:
        model_rgb = make_model_rgb(bt13, bt8)
        real_model_rgb.append(model_rgb)
        save_png(model_rgb, os.path.join(frames_real_dir, f"T{i+1:02d}_{ts}_MODEL_INPUT_TIR1WV.png"))
    else:
        real_model_rgb.append(None)

    print(f"  T{i+1:02d} [{ts}] composites:")
    composites = build_flood_composites(chans, out_prefix=f"T{i+1:02d}_REAL", ts_label=ts)
    real_flood_composites.append(composites)

# ==============================================================
#  STEP 2 - Level 1 interpolation (has ground truth + metrics,
#           identical logic to the base script)
# ==============================================================

interp_model_rgb = [None] * N
interp_flood_composites = [None] * N
all_metrics = []

# level1_frames: parallel lists (ts, bt13, bt8, is_real) representing
# the full level-1 sequence, real frames interleaved with their
# ground-truth-scored interpolated neighbors. This becomes the input
# sequence for hierarchy level 2 onward.
level1_ts = list(common_ts)
level1_bt13 = list(real_bt13)
level1_bt8 = list(real_bt8)
level1_is_real = [True] * N

if N >= 3:
    print(f"\n{'='*65}")
    print("Building LEVEL 1 interpolated sequence (has ground truth)...")

    interp_model_rgb[0] = real_model_rgb[0]
    interp_flood_composites[0] = real_flood_composites[0]

    for i in range(1, N - 1):
        ts_prev, ts_curr, ts_next = common_ts[i - 1], common_ts[i], common_ts[i + 1]

        can_run = (real_bt13[i - 1] is not None and real_bt8[i - 1] is not None and
                   real_bt13[i + 1] is not None and real_bt8[i + 1] is not None)

        if not can_run:
            print(f"  T{i+1:02d} -> SKIPPED (missing TIR1/WV on a neighbor frame), using real frame")
            interp_model_rgb[i] = real_model_rgb[i]
            interp_flood_composites[i] = real_flood_composites[i]
            level1_bt13[i] = real_bt13[i]
            level1_bt8[i] = real_bt8[i]
            continue

        half_gap_minutes = minutes_between(ts_prev, ts_next) / 2.0
        print(f"  T{i+1:02d} -> Model(T{i}, T{i+2})  gap={half_gap_minutes:.2f}min")

        pred = run_model(real_bt13[i - 1], real_bt8[i - 1], real_bt13[i + 1], real_bt8[i + 1])

        if pred is None:
            print(f"         Model failed -- falling back to real frame")
            interp_model_rgb[i] = real_model_rgb[i]
            interp_flood_composites[i] = real_flood_composites[i]
            level1_bt13[i] = real_bt13[i]
            level1_bt8[i] = real_bt8[i]
            continue

        interp_model_rgb[i] = pred
        save_png(pred, os.path.join(frames_interp_dir, f"T{i+1:02d}_{ts_curr}_MODEL_TIR1WV.png"))

        pred_bt13, pred_bt8 = rgb_to_bt(pred)

        # level 1's OWN prediction is what feeds hierarchy level 2+
        # (not the ground-truth real frame) -- this is intentional:
        # level 2 recurses on the model's actual output sequence.
        level1_bt13[i] = pred_bt13
        level1_bt8[i] = pred_bt8

        chans_pred = dict(all_channels_by_ts[ts_curr])
        chans_pred[MODEL_TIR_CHANNEL] = pred_bt13
        chans_pred[MODEL_WV_CHANNEL] = pred_bt8
        print(f"    Rebuilding flood composites at predicted T{i+1:02d} "
              f"(TIR1/WV predicted; other channels held from real frame):")
        interp_flood_composites[i] = build_flood_composites(
            chans_pred, out_prefix=f"T{i+1:02d}_PRED", ts_label=ts_curr)

        mm = compute_metrics(pred_bt13, pred_bt8, real_bt13[i], real_bt8[i])
        all_metrics.append({
            "frame_idx": i + 1, "label": f"T{i+1:02d}", "ts": ts_curr,
            "ts_prev": ts_prev, "ts_next": ts_next, "half_gap_min": half_gap_minutes,
            **{f"model_{k}": v for k, v in mm.items()},
        })
        print(f"         RMSE={mm['RMSE_avg']:.3f}K  PSNR={mm['PSNR_avg']:.2f}dB  "
              f"SSIM={mm['SSIM_avg']:.4f}  FSIM={mm['FSIM']:.4f}")

    interp_model_rgb[N - 1] = real_model_rgb[N - 1]
    interp_flood_composites[N - 1] = real_flood_composites[N - 1]

    # ---- Metrics CSV + plot (level 1 only -- the only level with GT) ----
    print(f"\n{'='*65}")
    print("Saving level-1 metrics (only level with ground truth)...")
    if all_metrics:
        import csv
        csv_path = os.path.join(metrics_dir, f"metrics_level1_T2_to_T{N-1}.csv")
        keys = list(all_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_metrics)
        print(f"  CSV -> {csv_path}")

        xs = [m["frame_idx"] for m in all_metrics]
        lbls = [m["label"] for m in all_metrics]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.patch.set_facecolor("#0d0d1a")
        fig.suptitle(f"Flood-test fine-tuned model Metrics (TIR1/WV) - Level 1 only - T2 to T{N-1}",
                     color="white", fontsize=13, fontweight="bold")
        plot_cfg = [
            (axes[0, 0], "RMSE avg (K)", "RMSE_avg"), (axes[0, 1], "PSNR avg (dB)", "PSNR_avg"),
            (axes[1, 0], "SSIM avg", "SSIM_avg"), (axes[1, 1], "FSIM", "FSIM"),
        ]
        for ax, title, key in plot_cfg:
            ax.set_facecolor("#16213e"); ax.tick_params(colors="white")
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xlabel("Frame", color="white", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            vals = [m.get(f"model_{key}", np.nan) for m in all_metrics]
            if not all(np.isnan(v) for v in vals):
                ax.plot(xs, vals, "o-", color="#2ecc71", linewidth=2, markersize=5)
            ax.set_xticks(xs); ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8, color="white")
            ax.yaxis.label.set_color("white")
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "metrics_plot_level1.png"), dpi=130,
                    bbox_inches="tight", facecolor="#0d0d1a")
        plt.close()
        print("  Metrics plot saved (level 1 only)")

    # ==============================================================
    #  STEP 3 - HIERARCHY LEVELS 2+ (no ground truth, purely for a
    #           smoother animation; each level recurses on the PREVIOUS
    #           level's frame sequence, real or synthetic alike)
    # ==============================================================

    if PROMPT_FOR_HIERARCHY_LEVELS:
        try:
            raw = input(
                f"\nHow many hierarchy levels? (1 = base interpolation only, "
                f"2 = one extra halving pass, 3 = two extra passes, ... ) "
                f"[default {HIERARCHY_LEVELS}]: "
            ).strip()
            n_levels = int(raw) if raw else HIERARCHY_LEVELS
        except Exception:
            print(f"  [WARN] Could not parse input, using default {HIERARCHY_LEVELS}.")
            n_levels = HIERARCHY_LEVELS
    else:
        n_levels = HIERARCHY_LEVELS

    n_levels = max(1, int(n_levels))
    print(f"\nRunning {n_levels} hierarchy level(s) total "
          f"({n_levels - 1} recursive halving pass(es) beyond level 1).")

    # current_* holds the deepest level built so far.
    current_ts = level1_ts
    current_bt13 = level1_bt13
    current_bt8 = level1_bt8
    current_is_real = level1_is_real

    level_frame_counts = [len(current_ts)]

    # all_levels_sequences: snapshot of EVERY level's full (ts, bt13, bt8,
    # is_real) frame lists, in order, level 1 first. This is what lets the
    # final animation show level 1 -> level 2 -> level 3 ... back to back
    # instead of only the deepest level. Level 1's snapshot is taken here;
    # each subsequent level's snapshot is appended at the end of its loop
    # iteration below.
    all_levels_sequences = [
        (1, list(current_ts), list(current_bt13), list(current_bt8), list(current_is_real))
    ]

    for level in range(2, n_levels + 1):
        print(f"\n{'='*65}")
        print(f"HIERARCHY LEVEL {level} -- no ground truth beyond this point, "
              f"metrics intentionally not computed.")
        n_cur = len(current_ts)
        if n_cur < 2:
            print(f"  [WARN] Only {n_cur} frame(s) at level {level - 1}; can't "
                  f"interpolate further. Stopping hierarchy early.")
            break

        new_ts = [current_ts[0]]
        new_bt13 = [current_bt13[0]]
        new_bt8 = [current_bt8[0]]
        new_is_real = [current_is_real[0]]

        for i in range(n_cur - 1):
            a_ts, b_ts = current_ts[i], current_ts[i + 1]
            a13, b13 = current_bt13[i], current_bt13[i + 1]
            a8, b8 = current_bt8[i], current_bt8[i + 1]

            mid_ts = midpoint_ts(a_ts, b_ts)

            if a13 is None or a8 is None or b13 is None or b8 is None:
                print(f"  [{level}] {a_ts} <-> {b_ts}: SKIPPED (missing TIR1/WV on a side)")
                mid13 = mid8 = None
            else:
                pred = run_model(a13, a8, b13, b8)
                if pred is None:
                    print(f"  [{level}] {a_ts} <-> {b_ts}: model failed, skipping midpoint")
                    mid13 = mid8 = None
                else:
                    mid13, mid8 = rgb_to_bt(pred)
                    save_png(pred, os.path.join(
                        hierarchy_dir, f"L{level}_{mid_ts}_MODEL_TIR1WV.png"))
                    print(f"  [{level}] {a_ts} <-> {b_ts} -> virtual {mid_ts} (no GT)")

            if mid13 is not None:
                new_ts.append(mid_ts)
                new_bt13.append(mid13)
                new_bt8.append(mid8)
                new_is_real.append(False)

            new_ts.append(b_ts)
            new_bt13.append(b13)
            new_bt8.append(b8)
            new_is_real.append(current_is_real[i + 1])

        current_ts, current_bt13, current_bt8, current_is_real = new_ts, new_bt13, new_bt8, new_is_real
        level_frame_counts.append(len(current_ts))
        all_levels_sequences.append(
            (level, list(current_ts), list(current_bt13), list(current_bt8), list(current_is_real))
        )
        print(f"  Level {level} done -- {len(current_ts)} frames "
              f"(was {level_frame_counts[-2]} at level {level - 1}).")

    print(f"\nFrame count by level: {level_frame_counts}")

    # ==============================================================
    #  STEP 4 - Build flood composites for EVERY level (1..n_levels),
    #           then render ONE combined animation that plays level 1's
    #           full sequence, then level 2's full sequence, then level
    #           3's, etc, back to back -- so you can watch it visibly
    #           get smoother as the hierarchy deepens, instead of only
    #           seeing the final deepest level in isolation.
    # ==============================================================

    print(f"\n{'='*65}")
    print(f"Building flood composites for all {len(all_levels_sequences)} level(s) "
          f"and rendering ONE combined animation...")

    def nearest_real_chans(ts, seq_ts, seq_is_real):
        """Other (non TIR1/WV) channels for a synthetic frame are
        borrowed from whichever REAL acquisition is closest in time and
        held constant -- same approach used at level 1, just applied at
        whatever level we're building composites for."""
        nearest_real_ts = min(
            (t for t, r in zip(seq_ts, seq_is_real) if r),
            key=lambda t: abs((parse_ts(t) - parse_ts(ts)).total_seconds())
        )
        return all_channels_by_ts[nearest_real_ts]

    # composites_by_level[level] = list of composite-dicts (or None),
    # one per frame in that level's sequence, same order as
    # all_levels_sequences.
    composites_by_level = {}
    for level, seq_ts, seq_bt13, seq_bt8, seq_is_real in all_levels_sequences:
        level_composites = []
        for i, ts in enumerate(seq_ts):
            bt13, bt8 = seq_bt13[i], seq_bt8[i]
            if bt13 is None or bt8 is None:
                level_composites.append(None)
                continue
            if seq_is_real[i]:
                chans_here = all_channels_by_ts[ts]
            else:
                chans_here = dict(nearest_real_chans(ts, seq_ts, seq_is_real))
                chans_here[MODEL_TIR_CHANNEL] = bt13
                chans_here[MODEL_WV_CHANNEL] = bt8
            prefix = "REAL" if seq_is_real[i] else f"L{level}_VIRTUAL"
            built = build_flood_composites(
                chans_here, out_prefix=f"COMBINED_{prefix}", ts_label=ts, quiet=True)
            level_composites.append(built)
        composites_by_level[level] = level_composites
        print(f"  Level {level}: composites built for {len(level_composites)} frame(s).")

    # Pick ONE composite key used throughout the whole combined GIF, so
    # every level renders the same way. Needs coverage across every
    # level's frames (falls back through FLOOD_COMPOSITES if
    # FEATURED_COMPOSITE isn't available everywhere).
    def has_full_coverage(key):
        for level_composites in composites_by_level.values():
            non_none = [c for c in level_composites if c is not None]
            if not non_none:
                continue
            if not all(key in c for c in non_none):
                return False
        return True

    chosen_key = FEATURED_COMPOSITE if has_full_coverage(FEATURED_COMPOSITE) else None
    if chosen_key is None:
        for key, desc, _ in FLOOD_COMPOSITES:
            if has_full_coverage(key):
                chosen_key = key
                break

    if chosen_key is None:
        print("  [skip] No single flood composite has coverage across all levels "
              "-- combined animation not built. Per-frame composite PNGs are "
              f"still available in {flood_composite_dir}/")
    else:
        print(f"  Using '{chosen_key}' composite for the combined animation "
              f"({'featured choice' if chosen_key == FEATURED_COMPOSITE else 'fallback, featured composite unavailable'}).")

        def make_single_frame(img, label, level, frame_num, total_frames_in_level, is_real):
            H, W = img.shape[:2]; HDR = 46
            canvas = np.full((H + HDR, W, 3), 18, dtype=np.uint8)
            canvas[HDR:HDR+H, 0:W] = (img * 255).astype(np.uint8)
            pil = Image.fromarray(canvas); draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, W, HDR - 1], fill=(10, 10, 22))
            try:
                font = ImageFont.truetype("arial.ttf", 14); font_sm = ImageFont.truetype("arial.ttf", 11)
            except Exception:
                font = ImageFont.load_default(); font_sm = font
            tag_color = (80, 210, 120) if is_real else (90, 160, 255)
            tag = "REAL" if is_real else "interp"
            draw.text((W // 2, HDR // 2 - 7), f"LEVEL {level}", fill=(255, 200, 60), font=font_sm, anchor="mm")
            draw.text((W // 2, HDR // 2 + 8), f"{label}  [{tag}]", fill=tag_color, font=font, anchor="mm")
            draw.text((W - 6, HDR - 6), f"{frame_num}/{total_frames_in_level}", fill=(130, 130, 130), font=font_sm, anchor="rb")
            return pil

        gif_frames = []
        duration_ms = int(1000 / GIF_FPS)
        for level, seq_ts, seq_bt13, seq_bt8, seq_is_real in all_levels_sequences:
            level_composites = composites_by_level[level]
            total_in_level = len(seq_ts)
            for i, ts in enumerate(seq_ts):
                comp = level_composites[i]
                if comp is None or chosen_key not in comp:
                    continue
                img = comp[chosen_key]
                frame_pil = make_single_frame(img, ts, level, i + 1, total_in_level, seq_is_real[i])
                if GIF_SCALE != 1.0:
                    nw = int(frame_pil.width * GIF_SCALE); nh = int(frame_pil.height * GIF_SCALE)
                    frame_pil = frame_pil.resize((nw, nh), Image.LANCZOS)
                gif_frames.append(frame_pil)
            print(f"  Level {level} added to combined animation "
                  f"({total_in_level} frame(s), running total {len(gif_frames)}).")

        if gif_frames:
            gif_path = os.path.join(
                anim_dir, f"flood_{chosen_key}_ALL_LEVELS_1_to_{n_levels}_combined.gif")
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                                duration=duration_ms, loop=0, optimize=False)
            print(f"\n  Combined animation ({chosen_key}, levels 1-{n_levels}, "
                  f"{len(gif_frames)} total frames across all levels) -> {gif_path}")

print(f"\n{'='*65}")
print("ALL DONE")
print(f"{'='*65}")
print(f"  Flood composite PNGs -> {flood_composite_dir}/")
print(f"  Model-input (TIR1/WV) real frames -> {frames_real_dir}/")
if N >= 3:
    print(f"  Level-1 interpolated frames -> {frames_interp_dir}/")
    print(f"  Level-1 metrics (ONLY level with ground truth) -> {metrics_dir}/")
    print(f"  Hierarchy levels 2+ frames (no ground truth) -> {hierarchy_dir}/")
    print(f"  Final smooth animation -> {anim_dir}/")
print()
print("Channel coverage summary:")
for i, ts in enumerate(common_ts):
    have = sorted(all_channels_by_ts[ts].keys())
    print(f"  T{i+1:02d} [{ts}]: {have}")
print()
print("Reminder: metrics (RMSE/PSNR/SSIM/FSIM) only exist for hierarchy level 1,")
print("where every interpolated frame has a real satellite acquisition to score")
print("against. Levels 2+ interpolate between frames that already include")
print("model-predicted TIR1/WV, so there is no ground truth to compare them to")
print("-- they exist purely to make the final animation smoother, and their")
print("accuracy is bounded (and degraded) by however good level 1 already was.")
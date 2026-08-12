"""
Day/Night Fog-Aware Frame Interpolation — Kaggle version (STREAMING / GPU)
============================================================================
[PORTED FOR KAGGLE FROM THE VERIFIED LOCAL DAY/NIGHT SCRIPT]

Same model architecture and day/night composite logic as the local
version (base_flownet + refine/ema_refine, 12-channel refine stem,
head_rgb, output_scale — verified against 6august.pth's real
state_dict).

CHANGES FROM THE PREVIOUS KAGGLE PORT
--------------------------------------
0. DAY/NIGHT MODE NOW DECIDED BY SOLAR ZENITH ANGLE, NOT ASTRAL
   SUNRISE/SUNSET. (unchanged from prior version — see get_mode())

1. GPU is now REQUIRED, not optional.

2. STREAMING SLIDING-WINDOW PROCESSING (fixes CPU OOM).

3. OUTPUT: only 3_comparisons/, 4_metrics/, 5_animation/ remain.

4. *** DAY-MODE COMPOSITE CHANGED (this patch) ***
   The day-mode RGB fed into the frozen IFNet flow-net used to be:
       R = VIS reflectance, G = SWIR reflectance, B = TIR1 BT
   A standalone diagnostic script (day_composite_comparison.py)
   swept 6 candidate channel arrangements across 18 real day-mode
   triplets spanning a night->day transition (2026-01-15, 08:00-
   16:30 IST) and scored each against ground truth with PSNR/SSIM/
   FSIM/LPIPS. Results (mean over 18 triplets):

       variant          PSNR   SSIM   FSIM   LPIPS
       current (old)    32.82  0.797  0.880  0.130
       A_tir_only       34.64  0.919  0.962  0.065   (rejected, see below)
       B_swir_vis_tir   33.21  0.808  0.897  0.128
       C_tir_vis_swir   33.49  0.811  0.895  0.125   <- adopted
       D_avg_tir_tir    34.29  0.885  0.938  0.078
       E_vis_gray       55.83  0.833  0.828  0.113   (metric artifact,
                                                        rejected)

   E_vis_gray's high mean PSNR is inflated by several afternoon
   triplets (high, stable sun angle) where grayscale VIS barely
   changes between neighbor frames, so trivial averaging nails the
   target almost exactly (PSNR 84+, SSIM=1.000, LPIPS=0.000) --
   that's a saturation artifact, not real interpolation quality.
   Rejected.

   A_tir_only scored best on raw accuracy metrics (best LPIPS/FSIM,
   most consistent near the night->day boundary), but it was
   REJECTED because the actual goal of this project is FOG
   detection/tracking, not just accuracy metrics. A_tir_only's day
   RGB is R=G=B=TIR1 (single band, tripled) -- it deletes VIS from
   the flow-net's RGB input entirely. VIS reflectance is the primary
   daytime fog/low-stratus discriminator (fog is bright/smooth in
   VIS relative to most backgrounds, especially morning fog) --
   this is the same reason night-mode already uses a TIR-MIR
   brightness-temperature difference as its fog signal instead of
   TIR alone. TIR by itself is well known to struggle separating fog
   from bare ground/clear sky, since both sit at similar brightness
   temperatures. So while A_tir_only interpolates smoother/more
   "plausible" frames by metrics, it does so by starving the flow
   estimator of the one band that actually carries the fog signal in
   daytime -- not an acceptable trade for this project's goal.

   C_tir_vis_swir was adopted instead: it keeps BOTH VIS and SWIR
   present in the flow-net's RGB input (so fog contrast is visible
   to motion/structure estimation, not just the refine-net's
   residual correction), while still giving the checkpoint a TIR
   anchor channel for partial in-distribution stability. It's a
   modest, more defensible improvement over "current" (PSNR
   32.82->33.49, LPIPS 0.130->0.125), including a large local win at
   the T22 11:00 mid-morning point (29.25->33.44 dB) -- not just at
   the day/night boundary extremes.

   WHAT CHANGED MECHANICALLY:
     - Day-mode RGB fed to the frozen IFNet flow-net is now
       R=TIR1 BT norm, G=VIS reflectance, B=SWIR reflectance
       (was R=VIS, G=SWIR, B=TIR).
     - tir0/tir1 side-channel (refine net) is unchanged: TIR1 BT
       normalized.
     - wv0/wv1 side-channel (refine net) is unchanged: SWIR
       reflectance (same as the old "current" composite).
     - Night-mode composite is COMPLETELY UNCHANGED (TIR/MIR/diff
       RGB, TIR tir-channel, MIR wv-channel).
     - This is still an inference-time mitigation, not a real fix:
       the checkpoint was trained on TIR/MIR (night-style) fields
       only, and no channel rearrangement changes that. A materially
       better day-mode result requires fine-tuning on real day-mode
       composites -- ideally including actual fog-labeled frames so
       the loss function can be told to care about fog boundary
       fidelity specifically, not just generic pixel-level PSNR/SSIM.

   NOT YET TESTED: a VIS-preserving, TIR-anchored variant like
   R=VIS, G=TIR, B=TIR (VIS gets full flow-net visibility, TIR
   duplicated for stability) was flagged as worth adding to the
   day_composite_comparison.py sweep but has not been run against
   real data yet. If you re-run that sweep, evaluate any winning
   variant on fog-visible frames specifically (not just aggregate
   PSNR/SSIM) before adopting it here, since this whole decision
   turned on a metric-vs-goal mismatch once already.

   The LOW-CONFIDENCE DAY-MODE GUARD (blending with real-neighbor
   average below ZENITH_RELIABLE_DAY_THRESHOLD) is KEPT as-is on
   top of this new composite. It was diagnosed against the old
   composite's low-sun-angle failure mode; C_tir_vis_swir is
   somewhat more stable near the boundary than "current" but not as
   dramatically as A_tir_only was, so this guard likely still
   matters here. Re-check ZENITH_RELIABLE_DAY_THRESHOLD against real
   runs before assuming it's still perfectly tuned.

DAY/NIGHT COMPOSITE LOGIC (UPDATED — day mode changed, see above)
-----------------------------------------------------------
  DAY MODE   (cos_z > ZENITH_DAY_THRESHOLD, per solar_zenith_cos()
              at a reference lat/lon):
      img0/img1 (RGB into frozen IFNet):
          R = TIR1 BT           (normalized 0-1)
          G = VIS reflectance   (0-1, TOA)
          B = SWIR reflectance  (0-1, TOA)
      tir0/tir1 (refine net):  TIR1 BT, normalized
      wv0/wv1   (refine net):  SWIR reflectance

  NIGHT MODE (cos_z <= ZENITH_DAY_THRESHOLD) -- unchanged:
      img0/img1 (RGB into frozen IFNet):
          R = TIR1 BT            (normalized 0-1)
          G = MIR1 BT            (normalized 0-1)
          B = (TIR1 - MIR1) diff (normalized 0-1)
      tir0/tir1 (refine net):  TIR1 BT, normalized
      wv0/wv1   (refine net):  MIR1 BT, normalized

CAVEAT (unchanged): best_checkpoint.pth was (presumably, same as
6august.pth) trained on TIR1+WV composites only. Feeding it VIS/
SWIR in the RGB slot is still an experimental probe -- shapes match
so it runs, but outputs for the VIS/SWIR channels specifically are
not guaranteed physically trustworthy without further validation or
fine-tuning. This composite prioritizes keeping the fog-relevant
bands (VIS/SWIR) visible to the model over raw interpolation
accuracy, per the project's actual goal (fog detection, not just
PSNR).

DAY/NIGHT BOUNDARY HANDLING (unchanged):
  When T[i-1] and T[i+1] fall in different modes, the script falls
  back to copying the real T[i] frame for that slot instead of
  running the model.

Input data format
------------------
INSAT-3D/3DR/3DS L1C .h5 files:
    IMG_TIR1 + IMG_TIR1_TEMP      (BT LUT, count->Kelvin)
    IMG_MIR  + IMG_MIR_TEMP       (BT LUT, count->Kelvin)
    IMG_VIS  + IMG_VIS_RADIANCE   (radiance LUT, count->radiance)
    IMG_SWIR + IMG_SWIR_RADIANCE  (radiance LUT, count->radiance)
"""

import os
import sys
import gc
import glob
import re
import copy
import math
import warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ==============================================================
#  SETTINGS - Kaggle paths
# ==============================================================

NC_FOLDER   = "/kaggle/input/datasets/divyashkigf/fognew15jan/fognewcrop15june"
OUTPUT_DIR  = "/kaggle/working/daynight_fog_outputveryverynew"

FINETUNED_CKPT_PATH = "/kaggle/input/datasets/mistridaivya/best-checkpoint-6th-aug/best_checkpoint.pth"
USE_EMA_WEIGHTS = True

INSAT_TIR_DATASET  = "IMG_TIR1"
INSAT_MIR_DATASET  = "IMG_MIR"
INSAT_VIS_DATASET  = "IMG_VIS"
INSAT_SWIR_DATASET = "IMG_SWIR"

REF_LAT = 22.5
REF_LON = 78.0
REF_TZ  = "Asia/Kolkata"

ZENITH_DAY_THRESHOLD = 0.05
ZENITH_RELIABLE_DAY_THRESHOLD = 0.70

BT_TIR1_MIN, BT_TIR1_MAX = 190.0, 310.0
BT_MIR_MIN,  BT_MIR_MAX  = 190.0, 330.0
BT_DIFF_MIN, BT_DIFF_MAX = -30.0, 30.0

VIS_REFL_MIN,  VIS_REFL_MAX  = 0.0, 1.0
SWIR_REFL_MIN, SWIR_REFL_MAX = 0.0, 1.0

ESUN_VIS  = 159.51661150792
ESUN_SWIR = 20.395574999586

GIF_FPS   = 2
GIF_SCALE = 1.0

REFINE_DROPOUT_P     = 0.08
REFINE_CHANNELS      = 64
REFINE_NUM_RESBLOCKS = 5

# ==============================================================
#  OUTPUT FOLDERS
# ==============================================================

compare_dir = os.path.join(OUTPUT_DIR, "3_comparisons")
metrics_dir = os.path.join(OUTPUT_DIR, "4_metrics")
anim_dir    = os.path.join(OUTPUT_DIR, "5_animation")

for d in [compare_dir, metrics_dir, anim_dir]:
    os.makedirs(d, exist_ok=True)

# ==============================================================
#  MODEL DEFINITIONS -- unchanged
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

if not torch.cuda.is_available():
    print("=" * 65)
    print("  ERROR: No GPU detected (torch.cuda.is_available() == False).")
    print("  Fix: Kaggle notebook -> Settings -> Accelerator -> pick a GPU,")
    print("  then save/restart the session and re-run this cell.")
    print("=" * 65)
    sys.exit(1)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"  device: {device}  ({torch.cuda.get_device_name(0)})")

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
        self.block_tea = IFBlock(10 + 4, c=90)

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
        in_ch = 3 + 4 + 1 + 1 + 1 + 1 + 1  # = 12
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
                "format. Got top-level keys: " +
                str(list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
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
#  GT-vs-INTERPOLATED METRICS: PSNR, SSIM, FSIM, LPIPS
# ==============================================================

print("Loading metric backends (skimage PSNR/SSIM, custom FSIM, LPIPS)...")

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import lpips as lpips_lib
except ImportError:
    print("  lpips not found -- installing...")
    os.system(f"{sys.executable} -m pip install -q lpips")
    import lpips as lpips_lib

_lpips_net = lpips_lib.LPIPS(net='alex').to(device)
_lpips_net.eval()
for p in _lpips_net.parameters():
    p.requires_grad_(False)
print("  LPIPS (AlexNet backbone) ready.")


def _log_gabor_phase_congruency(img_gray, nscale=4, norient=4, min_wavelength=6,
                                 mult=2.0, sigma_f=0.55, k=2.0):
    rows, cols = img_gray.shape
    imgfft = np.fft.fft2(img_gray)

    y, x = np.mgrid[-rows // 2:rows - rows // 2, -cols // 2:cols - cols // 2]
    y = y.astype(np.float64) / rows
    x = x.astype(np.float64) / cols
    radius = np.sqrt(x ** 2 + y ** 2)
    radius[rows // 2, cols // 2] = 1.0
    theta = np.arctan2(-y, x)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    total_energy = np.zeros((rows, cols), dtype=np.float64)
    total_amplitude_sum = np.zeros((rows, cols), dtype=np.float64) + 1e-4

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * norient / 2.0, np.pi)
        spread = (np.cos(dtheta) + 1.0) / 2.0

        sum_e = np.zeros((rows, cols), dtype=np.float64)
        sum_o = np.zeros((rows, cols), dtype=np.float64)
        an_array = []

        wavelength = min_wavelength
        for s in range(nscale):
            fo = 1.0 / wavelength
            log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_f) ** 2))
            log_gabor[rows // 2, cols // 2] = 0.0
            filt = np.fft.ifftshift(log_gabor * spread)

            eo = np.fft.ifft2(imgfft * filt)
            an = np.abs(eo)
            an_array.append(an)
            sum_e += eo.real
            sum_o += eo.imag
            wavelength *= mult

        xenergy = np.sqrt(sum_e ** 2 + sum_o ** 2) + 1e-4
        mean_e = sum_e / xenergy
        mean_o = sum_o / xenergy

        an_sum = np.zeros((rows, cols), dtype=np.float64)
        for an in an_array:
            an_sum += an

        wavelength = min_wavelength
        energy = np.zeros((rows, cols), dtype=np.float64)
        for s in range(nscale):
            fo = 1.0 / wavelength
            log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_f) ** 2))
            log_gabor[rows // 2, cols // 2] = 0.0
            filt = np.fft.ifftshift(log_gabor * spread)
            eo = np.fft.ifft2(imgfft * filt)
            energy += eo.real * mean_e + eo.imag * mean_o - np.abs(eo.real * mean_o - eo.imag * mean_e)
            wavelength *= mult

        energy = np.maximum(energy, 0.0)
        total_energy += energy
        total_amplitude_sum += an_sum

    pc = total_energy / total_amplitude_sum
    pc = np.clip(pc, 0.0, None)
    if pc.max() > 1e-8:
        pc = pc / (pc.max() + 1e-8)
    return pc


def _gradient_magnitude(img_gray):
    gx = np.gradient(img_gray, axis=1)
    gy = np.gradient(img_gray, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def fsim_single_channel(img1, img2, T1=0.85, T2=160.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    pc1 = _log_gabor_phase_congruency(img1)
    pc2 = _log_gabor_phase_congruency(img2)
    g1 = _gradient_magnitude(img1 * 255.0)
    g2 = _gradient_magnitude(img2 * 255.0)

    s_pc = (2 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
    s_g = (2 * g1 * g2 + T2) / (g1 ** 2 + g2 ** 2 + T2)
    s_l = s_pc * s_g

    pc_max = np.maximum(pc1, pc2)
    num = np.sum(s_l * pc_max)
    den = np.sum(pc_max) + 1e-8
    return float(num / den)


def compute_fsim_rgb(gt_rgb, pred_rgb):
    scores = []
    for c in range(3):
        try:
            scores.append(fsim_single_channel(gt_rgb[..., c], pred_rgb[..., c]))
        except Exception:
            scores.append(np.nan)
    return float(np.nanmean(scores))


def compute_lpips_rgb(gt_rgb, pred_rgb):
    def to_lpips_t(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        t = t * 2.0 - 1.0
        return t.to(device)
    with torch.no_grad():
        d = _lpips_net(to_lpips_t(gt_rgb), to_lpips_t(pred_rgb))
    return float(d.item())


def compute_all_metrics(gt_rgb, pred_rgb):
    gt_rgb = np.clip(gt_rgb, 0, 1).astype(np.float32)
    pred_rgb = np.clip(pred_rgb, 0, 1).astype(np.float32)

    try:
        psnr_val = sk_psnr(gt_rgb, pred_rgb, data_range=1.0)
    except Exception:
        psnr_val = float("nan")

    try:
        ssim_val = sk_ssim(gt_rgb, pred_rgb, data_range=1.0, channel_axis=2)
    except Exception:
        ssim_val = float("nan")

    try:
        fsim_val = compute_fsim_rgb(gt_rgb, pred_rgb)
    except Exception as e:
        print(f"    FSIM error: {e}")
        fsim_val = float("nan")

    try:
        lpips_val = compute_lpips_rgb(gt_rgb, pred_rgb)
    except Exception as e:
        print(f"    LPIPS error: {e}")
        lpips_val = float("nan")

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "fsim": fsim_val,
        "lpips": lpips_val,
    }

# ==============================================================
#  TIMESTAMP HANDLING
# ==============================================================

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def get_timestamp_dt(filepath):
    base = os.path.basename(filepath)
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_", base)
    if not m:
        return None
    dd, mon, yyyy, hhmm = m.groups()
    month = _MONTHS.get(mon.upper())
    if month is None:
        return None
    dd, hh, mm = int(dd), int(hhmm[:2]), int(hhmm[2:])
    return datetime(int(yyyy), month, dd, hh, mm, 0)

def minutes_between(dt_a, dt_b):
    return abs((dt_b - dt_a).total_seconds()) / 60.0

# ==============================================================
#  SOLAR GEOMETRY  (single source of truth for day/night + reflectance)
# ==============================================================

def earth_sun_distance_factor(dt: datetime) -> float:
    doy = dt.timetuple().tm_yday
    return 1.0 - 0.01672 * math.cos(2.0 * math.pi * (doy - 4) / 365.256)


def solar_zenith_cos(dt: datetime, lat: float, lon: float) -> float:
    dt_utc = dt - timedelta(hours=5, minutes=30)
    doy = dt_utc.timetuple().tm_yday

    decl = math.radians(23.45) * math.sin(math.radians(360.0 / 365.0 * (284 + doy)))
    frac_hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    solar_time = frac_hour + lon / 15.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))

    phi = math.radians(lat)
    cos_z = (math.sin(phi) * math.sin(decl) +
             math.cos(phi) * math.cos(decl) * math.cos(hour_angle))
    return cos_z


def radiance_to_reflectance(L, dt: datetime, lat: float, lon: float, esun: float) -> np.ndarray:
    d = earth_sun_distance_factor(dt)
    cos_z = solar_zenith_cos(dt, lat, lon)
    if cos_z <= 0.01:
        return np.zeros_like(L, dtype=np.float32)
    refl = (math.pi * L * (d ** 2)) / (esun * cos_z)
    return np.clip(np.nan_to_num(refl, nan=0.0), 0.0, 1.0).astype(np.float32)

# ==============================================================
#  DAY / NIGHT MODE DECISION  (solar zenith based, no astral)
# ==============================================================

_zenith_cache = {}

def get_mode(timestamp_ist: datetime) -> str:
    if timestamp_ist not in _zenith_cache:
        _zenith_cache[timestamp_ist] = solar_zenith_cos(timestamp_ist, REF_LAT, REF_LON)
    cos_z = _zenith_cache[timestamp_ist]
    return "day" if cos_z > ZENITH_DAY_THRESHOLD else "night"


def get_cos_z(timestamp_ist: datetime) -> float:
    if timestamp_ist not in _zenith_cache:
        _zenith_cache[timestamp_ist] = solar_zenith_cos(timestamp_ist, REF_LAT, REF_LON)
    return _zenith_cache[timestamp_ist]


def day_confidence_weight(cos_z: float) -> float:
    if cos_z >= ZENITH_RELIABLE_DAY_THRESHOLD:
        return 1.0
    if cos_z <= ZENITH_DAY_THRESHOLD:
        return 0.0
    return (cos_z - ZENITH_DAY_THRESHOLD) / (ZENITH_RELIABLE_DAY_THRESHOLD - ZENITH_DAY_THRESHOLD)

# ==============================================================
#  H5 LOADING HELPERS
# ==============================================================

def _load_h5_raw_and_lut(path, dataset_name):
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

        lut_name = f"{dataset_name}_TEMP"
        lut = f[lut_name][:].astype(np.float32) if lut_name in f else None

        radiance_lut = None
        rad_lut_name = f"{dataset_name}_RADIANCE"
        if rad_lut_name in f:
            radiance_lut = f[rad_lut_name][:].astype(np.float32)

    return raw, lut, radiance_lut, fill_value


def load_bt_band(path, dataset_name):
    raw, lut, _, fill_value = _load_h5_raw_and_lut(path, dataset_name)
    if lut is None:
        raise KeyError(f"No '{dataset_name}_TEMP' calibration LUT found in {path} "
                        f"-- cannot convert '{dataset_name}' counts to Kelvin.")
    idx = np.clip(raw, 0, lut.shape[0] - 1)
    bt = lut[idx]
    if fill_value is not None:
        bt = np.where(raw == fill_value, np.nan, bt)
    return bt.astype(np.float32)


def load_radiance_band(path, dataset_name):
    raw, _, radiance_lut, fill_value = _load_h5_raw_and_lut(path, dataset_name)
    if radiance_lut is None:
        raise KeyError(
            f"No '{dataset_name}_RADIANCE' LUT found in {path}. "
            f"Inspect the file's dataset keys and update load_radiance_band() "
            f"to match your actual VIS/SWIR calibration LUT name."
        )
    idx = np.clip(raw, 0, radiance_lut.shape[0] - 1)
    rad = radiance_lut[idx]
    if fill_value is not None:
        rad = np.where(raw == fill_value, np.nan, rad)
    return rad.astype(np.float32)

# ==============================================================
#  NORMALIZATION
# ==============================================================

def normalize(arr, vmin, vmax, invert=False):
    a = np.clip(arr, vmin, vmax)
    n = (a - vmin) / (vmax - vmin)
    return (1.0 - n) if invert else n

# ==============================================================
#  COMPOSITE BUILDERS
# ==============================================================
#
# *** PATCHED: build_day_inputs() now builds the "C_tir_vis_swir"
#     composite from the 18-triplet sweep — chosen over the raw
#     metric winner (A_tir_only) because A drops VIS entirely from
#     the flow-net RGB, and VIS reflectance is the primary daytime
#     fog/low-stratus signal this project actually needs. See header
#     notes for the full rationale and metric comparison. ***
#
#     OLD day RGB:  R=VIS reflectance, G=SWIR reflectance, B=TIR1 BT
#     NEW day RGB:  R=TIR1 BT, G=VIS reflectance, B=SWIR reflectance
#                   -- VIS and SWIR both stay in the flow-net's RGB
#                   input (so fog contrast is visible to motion/
#                   structure estimation), TIR moved to R as a partial
#                   in-distribution anchor for the checkpoint.
#
#     day wv-channel (refine net): UNCHANGED, still SWIR reflectance.
#     tir_norm side-channel: UNCHANGED, still TIR1 BT normalized.
#     Night mode (build_night_inputs) is completely untouched.

def build_day_inputs(path, dt):
    L_vis  = load_radiance_band(path, INSAT_VIS_DATASET)
    L_swir = load_radiance_band(path, INSAT_SWIR_DATASET)
    bt_tir = load_bt_band(path, INSAT_TIR_DATASET)

    refl_vis  = radiance_to_reflectance(L_vis,  dt, REF_LAT, REF_LON, ESUN_VIS)
    refl_swir = radiance_to_reflectance(L_swir, dt, REF_LAT, REF_LON, ESUN_SWIR)
    tir_norm  = normalize(bt_tir, BT_TIR1_MIN, BT_TIR1_MAX, invert=True)

    r = np.clip(refl_vis,  VIS_REFL_MIN,  VIS_REFL_MAX)   # kept for wv-channel below
    g = np.clip(refl_swir, SWIR_REFL_MIN, SWIR_REFL_MAX)  # kept for wv-channel below

    # NEW: day RGB is R=TIR, G=VIS, B=SWIR -- VIS/SWIR (the
    # fog-relevant bands) both stay in the flow-net's RGB input.
    rgb = np.dstack((tir_norm, refl_vis, refl_swir))
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

    return {
        "rgb": rgb,
        "tir_norm": np.nan_to_num(tir_norm, nan=0.0).astype(np.float32),
        "second_norm": np.nan_to_num(g, nan=0.0).astype(np.float32),  # SWIR, unchanged
        "mode": "day",
    }


def build_night_inputs(path, dt):
    bt_tir = load_bt_band(path, INSAT_TIR_DATASET)
    bt_mir = load_bt_band(path, INSAT_MIR_DATASET)

    tir_norm = normalize(bt_tir, BT_TIR1_MIN, BT_TIR1_MAX, invert=True)
    mir_norm = normalize(bt_mir, BT_MIR_MIN,  BT_MIR_MAX,  invert=True)
    diff = bt_tir - bt_mir
    diff_norm = normalize(diff, BT_DIFF_MIN, BT_DIFF_MAX, invert=False)

    r = tir_norm
    g = mir_norm
    b = diff_norm
    rgb = np.dstack((r, g, b))
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)

    return {
        "rgb": rgb,
        "tir_norm": np.nan_to_num(tir_norm, nan=0.0).astype(np.float32),
        "second_norm": np.nan_to_num(mir_norm, nan=0.0).astype(np.float32),
        "mode": "night",
    }


def build_inputs(path, dt):
    mode = get_mode(dt)
    if mode == "day":
        return build_day_inputs(path, dt)
    else:
        return build_night_inputs(path, dt)


# ==============================================================
#  BOUNDARY-SAFE COMPOSITE  (for day<->night transition pairs)
# ==============================================================
# TIR1 brightness temperature is the ONE band that means the same
# physical quantity in both day and night composites -- it's already
# R in day mode's RGB (as of the C_tir_vis_swir patch) and R in
# night mode's RGB, and it's the tir0/tir1 side-channel in both
# modes unconditionally. Everything else (VIS/SWIR reflectance vs
# MIR temperature / TIR-MIR diff) is NOT comparable across modes --
# averaging those would mix unrelated physical units into a
# meaningless composite, which would very likely be worse than the
# old real-frame fallback, not better.
#
# So for a day<->night boundary pair specifically, BOTH neighbor
# frames are rebuilt into a TIR-only RGB (R=G=B=TIR1 BT normalized,
# same construction as the earlier-rejected "A_tir_only" variant --
# rejected as the DEFAULT day composite because it throws away VIS/
# SWIR fog signal, but it's exactly the right tool here since a
# boundary pair has no shared fog-relevant channel to preserve in
# the first place). The wv-side-channel is set to a neutral 0.5
# (mid-gray) rather than borrowing either mode's real wv content,
# since neither VIS/SWIR (day) nor MIR (night) is meaningful for the
# other side of the pair.
#
# This lets the flow-net do REAL interpolation across the boundary
# using a physically consistent signal (thermal structure), instead
# of copying the real T[i] frame outright. It will not carry fog
# texture across the boundary (that information genuinely isn't
# present in a form valid on both sides), but it does propagate
# real thermal/cloud-top motion instead of freezing on a static
# real frame for 2 timesteps.

def build_boundary_inputs(inputs_any_mode):
    """Rebuild a TIR-only composite from an already-loaded frame's
    inputs dict (day or night), for use only at day<->night boundary
    pairs. Does not re-read the .h5 file -- reuses tir_norm already
    computed by build_day_inputs/build_night_inputs."""
    tir = inputs_any_mode["tir_norm"]
    rgb = np.dstack((tir, tir, tir)).astype(np.float32)
    neutral_wv = np.full_like(tir, 0.5, dtype=np.float32)
    return {
        "rgb": rgb,
        "tir_norm": tir,
        "second_norm": neutral_wv,
        "mode": inputs_any_mode["mode"] + "_boundary_tir",
    }

# ==============================================================
#  MODEL RUN (per frame pair)
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


def run_model(inputs_a, inputs_b):
    try:
        def to_t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        def band_to_t(arr2d):
            return torch.from_numpy(arr2d).unsqueeze(0).unsqueeze(0).float().to(device)

        img0 = to_t(inputs_a["rgb"])
        img1 = to_t(inputs_b["rgb"])
        tir0 = band_to_t(inputs_a["tir_norm"])
        tir1 = band_to_t(inputs_b["tir_norm"])
        wv0  = band_to_t(inputs_a["second_norm"])
        wv1  = band_to_t(inputs_b["second_norm"])

        img0, orig_hw = pad_to_multiple(img0, 32)
        img1, _       = pad_to_multiple(img1, 32)
        tir0, _       = pad_to_multiple(tir0, 32)
        tir1, _       = pad_to_multiple(tir1, 32)
        wv0,  _       = pad_to_multiple(wv0,  32)
        wv1,  _       = pad_to_multiple(wv1,  32)

        final, base_merged, residual = rife_model.inference(
            img0, img1, tir0, tir1, wv0, wv1, scale=1.0)

        final = unpad(final, orig_hw)
        pred = np.clip(final[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)

        del img0, img1, tir0, tir1, wv0, wv1, final, base_merged, residual
        torch.cuda.empty_cache()

        return pred

    except Exception as e:
        print(f"    Model error: {e}")
        torch.cuda.empty_cache()
        return None

# ==============================================================
#  COMPARISON PNG (single frame, transient in-memory images only)
# ==============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_comparison_png(real_img, pred_img, label_real, label_pred, idx, out_path,
                         metrics=None):
    diff = np.abs(real_img.astype(np.float32) - pred_img.astype(np.float32))
    diff_mag = diff.mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(f"{label_real}  |  {label_pred}", color="white", fontsize=10, fontweight="bold")

    for ax, img, lbl in [(axes[0], real_img, label_real), (axes[1], pred_img, label_pred)]:
        ax.set_facecolor("#0d0d1a")
        ax.imshow(img)
        ax.set_title(lbl, color="white", fontsize=8)
        ax.axis("off")

    axes[2].set_facecolor("#0d0d1a")
    im = axes[2].imshow(diff_mag, cmap="inferno", vmin=0, vmax=max(diff_mag.max(), 1e-6))
    diff_title = "Diff |Real - Pred|"
    if metrics is not None:
        diff_title += (f"\nPSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.3f}  "
                        f"FSIM={metrics['fsim']:.3f}  LPIPS={metrics['lpips']:.3f}")
    axes[2].set_title(diff_title, color="white", fontsize=7)
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white", labelsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)

# ==============================================================
#  GIF FRAME BUILDER (dual real/interp panel, from a comparison PNG)
# ==============================================================

from PIL import Image, ImageDraw, ImageFont

def make_dual_frame(real_img, interp_img, label_real, label_interp, frame_num, total_frames):
    H, W = real_img.shape[:2]
    GAP, HDR = 8, 44
    canvas = np.full((H + HDR, W * 2 + GAP * 3, 3), 18, dtype=np.uint8)
    canvas[HDR:HDR+H, GAP:GAP+W] = (real_img * 255).astype(np.uint8)
    canvas[HDR:HDR+H, GAP+W+GAP:GAP+W+GAP+W] = (interp_img * 255).astype(np.uint8)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, W*2 + GAP*3, HDR - 1], fill=(10, 10, 22))
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    draw.text((GAP + W // 2, HDR // 2), label_real, fill=(80, 210, 120), font=font, anchor="mm")
    draw.text((GAP + W + GAP + W // 2, HDR // 2), label_interp, fill=(90, 160, 255), font=font, anchor="mm")
    draw.text((W*2 + GAP*3 - 6, HDR - 6), f"{frame_num}/{total_frames}",
              fill=(130, 130, 130), font=font_sm, anchor="rb")
    x_div = GAP + W + GAP // 2
    draw.line([(x_div, HDR), (x_div, H + HDR)], fill=(50, 50, 60), width=2)
    return pil

# ==============================================================
#  SCAN & MATCH FILES
# ==============================================================

print(f"\nScanning data folder ({NC_FOLDER})...")
all_h5 = glob.glob(os.path.join(NC_FOLDER, "*.h5")) + glob.glob(os.path.join(NC_FOLDER, "*.hdf5"))

frame_list = []
for f in all_h5:
    dt = get_timestamp_dt(f)
    if dt is not None:
        frame_list.append((dt, f))
frame_list.sort(key=lambda x: x[0])

print(f"  .h5 files found: {len(all_h5)}  Parsed timestamps: {len(frame_list)}")

if len(frame_list) < 3:
    print("ERROR: Need at least 3 timestamped .h5 files.")
    sys.exit(1)

N = len(frame_list)
print(f"  Using {N} frames  (zenith day/night threshold: cos_z > {ZENITH_DAY_THRESHOLD}):")
for i, (dt, path) in enumerate(frame_list):
    cz = solar_zenith_cos(dt, REF_LAT, REF_LON)
    print(f"    T{i+1}: {dt.strftime('%Y-%m-%d %H:%M')} IST  [{get_mode(dt).upper()}]  "
          f"cos_z={cz:+.3f}  {os.path.basename(path)}")

# ==============================================================
#  STREAMING SLIDING-WINDOW PROCESSING
# ==============================================================

print(f"\n{'='*65}")
print("Streaming sliding-window pass (load -> interp -> compare -> free)...")

all_records = []
gif_frames = []

loaded_cache = {}

def get_loaded(i):
    if i not in loaded_cache:
        dt_i, path_i = frame_list[i]
        loaded_cache[i] = build_inputs(path_i, dt_i)
    return loaded_cache[i]

def drop_loaded(i):
    if i in loaded_cache:
        del loaded_cache[i]

_BOUNDARY_METRICS = {"psnr": float("inf"), "ssim": 1.0, "fsim": 1.0, "lpips": 0.0}

dt0, path0 = frame_list[0]
inp0 = get_loaded(0)
gif_frames.append(make_dual_frame(inp0["rgb"], inp0["rgb"],
                                   f"T01 REAL [{inp0['mode']}]", "T01 (real, boundary)", 1, N))
save_comparison_png(inp0["rgb"], inp0["rgb"],
                     f"T01 REAL [{inp0['mode']}]  {dt0.strftime('%H:%M')}",
                     "T01 REAL (boundary)",
                     0, os.path.join(compare_dir, "T01_compare.png"),
                     metrics=_BOUNDARY_METRICS)
all_records.append({
    "frame_idx": 1, "ts": dt0.strftime("%Y-%m-%d %H:%M"), "mode": inp0["mode"],
    "half_gap_min": 0.0, "fallback": True, "low_confidence": False, "blend_weight": "",
    **_BOUNDARY_METRICS,
})
print(f"  T01 -> real (boundary, no T0)")

for i in range(1, N - 1):
    dt_prev, _ = frame_list[i - 1]
    dt_curr, path_curr = frame_list[i]
    dt_next, _ = frame_list[i + 1]

    inp_prev = get_loaded(i - 1)
    inp_next = get_loaded(i + 1)

    mode_prev = inp_prev["mode"]
    mode_next = inp_next["mode"]
    half_gap_minutes = minutes_between(dt_prev, dt_next) / 2.0

    print(f"  T{i+1:02d} -> Model(T{i}[{mode_prev}], T{i+2}[{mode_next}])  gap={half_gap_minutes:.2f}min")

    inp_curr = get_loaded(i)

    if mode_prev != mode_next:
        print(f"         Day/night boundary pair -- interpolating in shared TIR-only space")
        bound_prev = build_boundary_inputs(inp_prev)
        bound_next = build_boundary_inputs(inp_next)
        pred = run_model(bound_prev, bound_next)
        if pred is None:
            print(f"         Boundary model failed -- falling back to real frame")
            pred_rgb = inp_curr["rgb"]
            pred_mode = inp_curr["mode"]
            fallback = True
            low_confidence = False
            blend_weight = None
        else:
            # pred is a TIR-only (grayscale-in-RGB) interpolated
            # frame. Compare it against a TIR-only rendering of the
            # real T[i] frame too, since the real T[i] frame is in
            # its own (day or night) RGB space and isn't directly
            # comparable to this TIR-only prediction pixel-for-pixel.
            pred_rgb = pred
            pred_mode = inp_curr["mode"] + "_boundary_tir"
            fallback = False
            low_confidence = True   # still flagged: fog/VIS/MIR content
                                     # is NOT present in this prediction,
                                     # only TIR-derived structure/motion.
            blend_weight = None
    else:
        pred = run_model(inp_prev, inp_next)
        if pred is None:
            print(f"         Model failed -- falling back to real frame")
            pred_rgb = inp_curr["rgb"]
            pred_mode = inp_curr["mode"]
            fallback = True
            low_confidence = False
            blend_weight = None
        else:
            fallback = False
            pred_mode = mode_prev

            if pred_mode == "day":
                cos_z_curr = get_cos_z(dt_curr)
                w = day_confidence_weight(cos_z_curr)
                if w < 1.0:
                    neighbor_avg = (inp_prev["rgb"].astype(np.float32) +
                                     inp_next["rgb"].astype(np.float32)) / 2.0
                    pred_rgb = np.clip(w * pred + (1.0 - w) * neighbor_avg, 0, 1).astype(np.float32)
                    low_confidence = True
                    blend_weight = w
                    print(f"         done [day, LOW-CONFIDENCE cos_z={cos_z_curr:.3f} "
                          f"-> blend weight(model)={w:.2f}]")
                else:
                    pred_rgb = pred
                    low_confidence = False
                    blend_weight = 1.0
                    print(f"         done [day, cos_z={cos_z_curr:.3f}, full confidence]")
            else:
                pred_rgb = pred
                low_confidence = False
                blend_weight = 1.0
                print(f"         done [{mode_prev}]")

    if fallback:
        frame_metrics = dict(_BOUNDARY_METRICS)
    elif mode_prev != mode_next:
        # Boundary TIR-only prediction: comparing it against the
        # full-color real frame directly isn't meaningful (different
        # color spaces -- pred has no VIS/SWIR/MIR content by
        # construction). Score against a TIR-only rendering of the
        # real frame instead, for a fair like-for-like comparison.
        real_tir_only = build_boundary_inputs(inp_curr)["rgb"]
        frame_metrics = compute_all_metrics(real_tir_only, pred_rgb)
        print(f"         [scored vs TIR-only real, not full-color real]  "
              f"PSNR={frame_metrics['psnr']:.2f}  SSIM={frame_metrics['ssim']:.3f}  "
              f"FSIM={frame_metrics['fsim']:.3f}  LPIPS={frame_metrics['lpips']:.3f}")
    else:
        frame_metrics = compute_all_metrics(inp_curr["rgb"], pred_rgb)
        print(f"         PSNR={frame_metrics['psnr']:.2f}  SSIM={frame_metrics['ssim']:.3f}  "
              f"FSIM={frame_metrics['fsim']:.3f}  LPIPS={frame_metrics['lpips']:.3f}")

    label_real = f"T{i+1:02d} REAL [{inp_curr['mode']}]  {dt_curr.strftime('%H:%M')}"
    if fallback and mode_prev != mode_next:
        label_pred = f"T{i+1:02d} REAL (boundarymode)"
    elif fallback:
        label_pred = f"T{i+1:02d} REAL (fallback)"
    elif mode_prev != mode_next:
        label_pred = f"T{i+1:02d} MODEL [TIR-only boundary interp]  {dt_curr.strftime('%H:%M')}"
    elif low_confidence:
        label_pred = f"T{i+1:02d} MODEL+BLEND [{pred_mode}] w={blend_weight:.2f}  {dt_curr.strftime('%H:%M')}"
    else:
        label_pred = f"T{i+1:02d} MODEL [{pred_mode}]  {dt_curr.strftime('%H:%M')}"

    save_comparison_png(inp_curr["rgb"], pred_rgb, label_real, label_pred, i,
                         os.path.join(compare_dir, f"T{i+1:02d}_compare.png"),
                         metrics=frame_metrics)

    gif_frames.append(make_dual_frame(
        inp_curr["rgb"], pred_rgb,
        f"T{i+1:02d} REAL [{inp_curr['mode']}]",
        f"T{i+1:02d} {'(fallback)' if fallback else ('(TIR-boundary)' if mode_prev != mode_next else ('(model+blend)' if low_confidence else '(model)'))} [{pred_mode}]",
        i + 1, N))

    all_records.append({
        "frame_idx": i + 1,
        "ts": dt_curr.strftime("%Y-%m-%d %H:%M"),
        "mode": pred_mode,
        "half_gap_min": half_gap_minutes,
        "fallback": fallback,
        "low_confidence": low_confidence,
        "blend_weight": blend_weight if blend_weight is not None else "",
        **frame_metrics,
    })

    drop_loaded(i - 1)
    drop_loaded(i)
    gc.collect()

dt_last, path_last = frame_list[N - 1]
inp_last = get_loaded(N - 1)
gif_frames.append(make_dual_frame(inp_last["rgb"], inp_last["rgb"],
                                   f"T{N:02d} REAL [{inp_last['mode']}]", f"T{N:02d} (real, boundary)", N, N))
save_comparison_png(inp_last["rgb"], inp_last["rgb"],
                     f"T{N:02d} REAL [{inp_last['mode']}]  {dt_last.strftime('%H:%M')}",
                     f"T{N:02d} REAL (boundary)",
                     N - 1, os.path.join(compare_dir, f"T{N:02d}_compare.png"),
                     metrics=_BOUNDARY_METRICS)
all_records.append({
    "frame_idx": N, "ts": dt_last.strftime("%Y-%m-%d %H:%M"), "mode": inp_last["mode"],
    "half_gap_min": 0.0, "fallback": True, "low_confidence": False, "blend_weight": "",
    **_BOUNDARY_METRICS,
})
drop_loaded(N - 1)
gc.collect()
print(f"  T{N:02d} -> real (boundary, no T{N+1})")

print(f"  {N} comparison PNGs saved -> {compare_dir}/")

# ==============================================================
#  RECORDS CSV
# ==============================================================

print(f"\n{'='*65}")
print("Saving records...")

all_records.sort(key=lambda r: r["frame_idx"])

if all_records:
    import csv
    csv_path = os.path.join(metrics_dir, f"records_T01_to_T{N:02d}.csv")
    keys = list(all_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)
    print(f"  CSV -> {csv_path}")

    real_pred_records = [r for r in all_records if not r["fallback"]]
    if real_pred_records:
        finite_psnr = [r["psnr"] for r in real_pred_records if math.isfinite(r["psnr"])]
        print(f"\n  Model-interpolated frames: {len(real_pred_records)} / {len(all_records)}")
        if finite_psnr:
            print(f"    PSNR  mean={np.mean(finite_psnr):.2f}  min={np.min(finite_psnr):.2f}  max={np.max(finite_psnr):.2f}")
        print(f"    SSIM  mean={np.mean([r['ssim'] for r in real_pred_records]):.4f}")
        print(f"    FSIM  mean={np.mean([r['fsim'] for r in real_pred_records]):.4f}")
        print(f"    LPIPS mean={np.mean([r['lpips'] for r in real_pred_records]):.4f}  (lower = better)")

        lc_records = [r for r in real_pred_records if r["low_confidence"]]
        if lc_records:
            lc_psnr = [r["psnr"] for r in lc_records if math.isfinite(r["psnr"])]
            print(f"\n  Low-confidence day-mode frames (blended, cos_z < {ZENITH_RELIABLE_DAY_THRESHOLD}): "
                  f"{len(lc_records)} / {len(real_pred_records)}")
            if lc_psnr:
                print(f"    PSNR  mean={np.mean(lc_psnr):.2f}  min={np.min(lc_psnr):.2f}  max={np.max(lc_psnr):.2f}")
            print(f"    (these are mitigated via model/real-neighbor blending -- see "
                  f"ZENITH_RELIABLE_DAY_THRESHOLD comment; this is NOT a substitute for "
                  f"fine-tuning the model on day-mode composites)")

# ==============================================================
#  METRICS PLOT (PSNR / SSIM / FSIM / LPIPS vs frame index)
# ==============================================================

print(f"\n{'='*65}")
print("Plotting metrics...")

if all_records:
    idxs = [r["frame_idx"] for r in all_records]
    is_fb = [r["fallback"] for r in all_records]

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.patch.set_facecolor("#0d0d1a")

    metric_specs = [
        ("psnr", "PSNR (dB, higher=better)", "#5fd48c"),
        ("ssim", "SSIM (higher=better)", "#5aa0ff"),
        ("fsim", "FSIM (higher=better)", "#f0c05a"),
        ("lpips", "LPIPS (lower=better)", "#ff6f6f"),
    ]

    for ax, (key, title, color) in zip(axs, metric_specs):
        vals = [r[key] for r in all_records]
        vals_plot = [np.nan if (key == "psnr" and not math.isfinite(v)) else v for v in vals]

        ax.set_facecolor("#0d0d1a")
        ax.plot(idxs, vals_plot, marker="o", markersize=4, color=color, linewidth=1.5,
                label="model-interpolated")

        fb_idxs = [i for i, f in zip(idxs, is_fb) if f]
        for fi in fb_idxs:
            ax.axvline(fi, color="#888888", linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_title(title, color="white", fontsize=10, loc="left")
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#333344", linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#444455")

    axs[-1].set_xlabel("Frame index (T)", color="white", fontsize=9)
    fig.suptitle("GT vs Interpolated -- per-frame metrics\n"
                  "(dotted vertical lines = boundary/fallback frames, trivial perfect match)",
                  color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    metrics_plot_path = os.path.join(metrics_dir, "metrics_plot.png")
    plt.savefig(metrics_plot_path, dpi=130, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close(fig)
    print(f"  Metrics plot -> {metrics_plot_path}")

# ==============================================================
#  GIF (built from the frames accumulated during the streaming pass)
# ==============================================================

print(f"\n{'='*65}")
print("Building dual animation GIF...")

duration_ms = int(1000 / GIF_FPS)

if GIF_SCALE != 1.0:
    gif_frames = [
        fp.resize((int(fp.width * GIF_SCALE), int(fp.height * GIF_SCALE)), Image.LANCZOS)
        for fp in gif_frames
    ]

gif_path = os.path.join(anim_dir, "daynight_real_vs_interpolated.gif")
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                    duration=duration_ms, loop=0, optimize=False)
print(f"\n  GIF -> {gif_path}")

gif_slow = os.path.join(anim_dir, "daynight_real_vs_interpolated_slow.gif")
gif_frames[0].save(gif_slow, save_all=True, append_images=gif_frames[1:],
                    duration=800, loop=0, optimize=False)
print(f"  Slow GIF -> {gif_slow}")

# ==============================================================
#  SUMMARY
# ==============================================================

print(f"\n{'='*65}")
print("ALL DONE")
print(f"{'='*65}")
print(f"  Comparison PNGs (Real | Pred | Diff) -> {compare_dir}/")
print(f"  Records CSV + metrics plot           -> {metrics_dir}/")
print(f"  GIFs                                 -> {anim_dir}/")
fallback_count = sum(1 for r in all_records if r["fallback"])
print(f"  Day/night boundary/fallback frames: {fallback_count} / {len(all_records)} total frames")
print(f"{'='*65}")
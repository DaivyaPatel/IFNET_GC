#!/usr/bin/env python3
"""
RIFE Evolution Fine-Tune v5.4 — RANDOM INIT + TURBO (FINAL)
===============================================================
v5 (zero-init) proved the architecture is safe but paralyzed.
v5.4 escapes the zero basin with random init + turbo config.

USAGE
-----
    python rife_evolution_v5_4_random_turbo.py --diagnostic
    python rife_evolution_v5_4_random_turbo.py --cleaned-train-list tri_trainlist_cleaned.txt --cleaned-val-list tri_testlist_cleaned.txt
"""

import os
import sys
import csv
import time
import copy
import random
import warnings
import shutil
import gc
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

KAGGLE_MODE = False

BASE_DIR        = Path(__file__).resolve().parent
DATA_ROOT_SEQ   = str(BASE_DIR)
DATA_ROOT_LOGS  = str(BASE_DIR / "bah" / "finetune-logs")
RIFE_FOLDER     = str(BASE_DIR / "bah" / "model")
PRETRAIN_PATH   = os.path.join(RIFE_FOLDER, "train_log")
OUTPUT_DIR      = str(BASE_DIR / "rife_finetuned")

CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

DIAGNOSTIC_MODE = False
DIAG_SUBSET_FRACTION = 0.20
DIAG_EPOCHS          = 10
DIAG_BATCH_SIZE      = 8
DIAG_GRAD_ACCUM      = 2

BT13_MIN, BT13_MAX = 190.0, 310.0
BT8_MIN,  BT8_MAX  = 190.0, 280.0

FREEZE_BASE_FLOWNET   = True
REFINE_LR             = 5e-5
REFINE_WEIGHT_DECAY   = 1e-4
REFINE_DROPOUT_P      = 0.08
REFINE_CHANNELS       = 64
REFINE_NUM_RESBLOCKS  = 5

BOOTSTRAP_EPOCHS      = 2
BOOTSTRAP_LR          = 2e-4

EMA_DECAY              = 0.995
W_CHARBONNIER          = 1.0
W_BAND_CHARBONNIER_TIR = 0.5
W_BAND_CHARBONNIER_WV  = 1.2
W_SSIM                 = 0.1
W_GRADIENT             = 0.40

OUTPUT_SCALE_TIR = 0.25
OUTPUT_SCALE_WV  = 0.10
OUTPUT_SCALE_B   = 0.15

GATE_GAP_LOW_MIN  = 20.0
GATE_GAP_HIGH_MIN = 30.0

TRAIN_GAP_FILTER = None

BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 4
NUM_WORKERS      = 4
PIN_MEMORY       = True
NUM_EPOCHS       = 50
WARMUP_EPOCHS    = 2
GRAD_CLIP        = 0.35

ADAMW_BETA2                = 0.95
SPIKE_LOSS_MULTIPLIER      = 8.0
SPIKE_EMA_DECAY            = 0.98
SPIKE_WARMUP_STEPS         = 20
EPOCH_DIVERGENCE_RATIO     = 4.0
SANE_LOSS_CEILING_MULTIPLIER = 3.0
MAX_CONSECUTIVE_EXTREME_GRAD = 5
MAX_MID_EPOCH_ROLLBACKS      = 5

VAL_EVERY        = 1
KEEP_LAST_N      = 3
EARLY_STOP_ENABLED   = True
EARLY_STOP_PATIENCE  = 15
TRAIN_VAL_GAP_WARN_DB = 3.0

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log_ram(tag=""):
    try:
        import psutil
        mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"  [RAM {tag}] {mb:.0f} MB")
    except ImportError:
        pass

def aggressive_gc():
    gc.set_threshold(700, 10, 10)
    gc.collect(); gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True)

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
    """
    [v5.4 — RANDOM INIT ESCAPE]
    v5 (zero-init) was safe but paralyzed. v5.4 changes ONLY the init:
      - head_rgb weight: random normal std=0.005 (was zeros)
      - Diagnostic subset: 20% (was 5%)
      - Bootstrap: 5 epochs @ 5e-4 (was 2 @ 2e-4)
      - EMA: 0.995 (was 0.999)
      - W_GRADIENT: 0.40 (was 0.15)
      - OUTPUT_SCALE_TIR: 0.25 (was 0.15)
    The tanh bound + gap gate architecture is unchanged and proven safe.
    """
    def __init__(self, channels=REFINE_CHANNELS, n_resblocks=REFINE_NUM_RESBLOCKS,
                 dropout_p=REFINE_DROPOUT_P):
        super().__init__()
        in_ch = 3 + 4 + 1 + 1 + 1 + 1 + 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, 1, 1),
            nn.PReLU(channels),
        )
        self.resblocks = nn.Sequential(*[ResBlock(channels, dropout_p) for _ in range(n_resblocks)])
        self.head_rgb = nn.Conv2d(channels, 3, 3, 1, 1)
        # [CHANGED v5.4] RANDOM INIT to escape zero basin
        nn.init.normal_(self.head_rgb.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.head_rgb.bias)
        self.register_buffer(
            "output_scale",
            torch.tensor([OUTPUT_SCALE_TIR, OUTPUT_SCALE_WV, OUTPUT_SCALE_B]).view(1, 3, 1, 1))

    def forward(self, base_merged, flow, tir0, tir1, wv0, wv1, gap):
        H, W = base_merged.shape[2], base_merged.shape[3]
        gap_map = gap.view(-1, 1, 1, 1).expand(-1, 1, H, W)
        x = torch.cat([base_merged, flow, tir0, tir1, wv0, wv1, gap_map], dim=1)
        feat = self.stem(x)
        feat = self.resblocks(feat)
        raw = self.head_rgb(feat)
        delta = torch.tanh(raw) * self.output_scale
        return delta


class Model:
    def __init__(self):
        self.base_flownet = IFNet().to(device)
        self.refine = EvolutionRefinementNet().to(device)
        self.ema_refine = copy.deepcopy(self.refine).to(device)
        for p in self.ema_refine.parameters():
            p.requires_grad_(False)
        self._refine_forward_compiled = self.refine

        if FREEZE_BASE_FLOWNET:
            for p in self.base_flownet.parameters():
                p.requires_grad_(False)
            self.base_flownet.eval()

    def train(self):
        if not FREEZE_BASE_FLOWNET:
            self.base_flownet.train()
        self.refine.train()

    def eval(self):
        self.base_flownet.eval()
        self.refine.eval()

    def load_pretrained_base(self, path):
        ckpt_path = os.path.join(path, "flownet.pkl") if os.path.isdir(path) else path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        missing, unexpected = self.base_flownet.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"  [WARN] base_flownet load — missing: {len(missing)}, unexpected: {len(unexpected)}")
        else:
            print("  base_flownet loaded with ZERO missing/unexpected keys — exact match.")
        del ckpt
        aggressive_gc()

    def forward(self, img0, img1, tir0, tir1, wv0, wv1, gap, scale_list=(4, 2, 1),
                use_base_only=False):
        base_ctx = torch.no_grad() if FREEZE_BASE_FLOWNET else torch.enable_grad()
        with base_ctx:
            x = torch.cat((img0, img1), 1)
            flow_list, mask, merged = self.base_flownet(x, scale_list=scale_list)
        base_flow = flow_list[2]
        base_mask = mask
        base_merged = merged[2]

        with torch.no_grad():
            base_tir_merged = warp(tir0, base_flow[:, :2]) * base_mask + warp(tir1, base_flow[:, 2:4]) * (1 - base_mask)
            base_wv_merged  = warp(wv0,  base_flow[:, :2]) * base_mask + warp(wv1,  base_flow[:, 2:4]) * (1 - base_mask)

        if use_base_only:
            return base_merged, base_merged, None, base_flow, base_mask, base_tir_merged, base_wv_merged

        final_delta = self._refine_forward_compiled(
            base_merged.detach() if FREEZE_BASE_FLOWNET else base_merged,
            base_flow.detach() if FREEZE_BASE_FLOWNET else base_flow,
            tir0, tir1, wv0, wv1, gap)

        gate_low = GATE_GAP_LOW_MIN / 60.0
        gate_high = GATE_GAP_HIGH_MIN / 60.0
        gate = torch.clamp((gap - gate_low) / (gate_high - gate_low), 0.0, 1.0)
        final_delta = final_delta * gate.view(-1, 1, 1, 1)

        final_rgb = torch.clamp(base_merged + final_delta, 0.0, 1.0)

        return final_rgb, base_merged, final_delta, base_flow, base_mask, base_tir_merged, base_wv_merged

    def inference(self, img0, img1, tir0, tir1, wv0, wv1, gap=0.5, scale=1.0):
        if isinstance(gap, float):
            gap = torch.tensor([gap], dtype=torch.float32, device=img0.device)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            final, base, _, flow, mask, _, _ = self.forward(
                img0, img1, tir0, tir1, wv0, wv1, gap, scale_list=scale_list)
        return final, base, flow

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_refine.parameters(), self.refine.parameters()):
                ema_p.mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)
            for ema_b, b in zip(self.ema_refine.buffers(), self.refine.buffers()):
                ema_b.copy_(b)

    def save_refine(self, path, use_ema=False):
        os.makedirs(path, exist_ok=True)
        state = (self.ema_refine if use_ema else self.refine).state_dict()
        torch.save(state, os.path.join(path, "refine_state.pth"))


def gaussian_window(size, sigma, channel, dev):
    coords = torch.arange(size, dtype=torch.float32, device=dev) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w2d = g.view(1, 1, size, 1) * g.view(1, 1, 1, size)
    return w2d.expand(channel, 1, size, size).contiguous()

def ssim_loss(img1, img2, ws=11, C1=1e-4, C2=9e-4):
    ch = img1.size(1)
    win = gaussian_window(ws, 1.5, ch, img1.device)
    mu1 = F.conv2d(img1, win, padding=ws // 2, groups=ch)
    mu2 = F.conv2d(img2, win, padding=ws // 2, groups=ch)
    s1 = F.conv2d(img1 * img1, win, padding=ws // 2, groups=ch) - mu1 ** 2
    s2 = F.conv2d(img2 * img2, win, padding=ws // 2, groups=ch) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, win, padding=ws // 2, groups=ch) - mu1 * mu2
    return 1.0 - ((2 * mu1 * mu2 + C1) * (2 * s12 + C2) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()

def charbonnier(pred, gt, eps=1e-6):
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps ** 2))

def gradient_loss(pred, gt):
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    return torch.mean(torch.abs(pred_dy - gt_dy)) + torch.mean(torch.abs(pred_dx - gt_dx))

def compute_losses(final_rgb, gt_rgb):
    lc_rgb = charbonnier(final_rgb, gt_rgb)
    lc_tir = charbonnier(final_rgb[:, 0:1, :, :], gt_rgb[:, 0:1, :, :])
    lc_wv  = charbonnier(final_rgb[:, 1:2, :, :], gt_rgb[:, 1:2, :, :])
    ls = ssim_loss(final_rgb, gt_rgb)
    lg = gradient_loss(final_rgb, gt_rgb)

    total = (W_CHARBONNIER * lc_rgb +
             W_BAND_CHARBONNIER_TIR * lc_tir + W_BAND_CHARBONNIER_WV * lc_wv +
             W_SSIM * ls + W_GRADIENT * lg)

    logs = {
        "charb_rgb": lc_rgb.item(), "charb_tir": lc_tir.item(), "charb_wv": lc_wv.item(),
        "ssim": ls.item(), "grad": lg.item(), "total": total.item()
    }
    return total, logs

def calculate_psnr(p, g):
    mse = torch.mean((p - g) ** 2)
    if mse < 1e-10:
        return 100.0
    return (20 * torch.log10(torch.tensor(1.0, device=mse.device) / torch.sqrt(mse))).item()

def calculate_psnr_per_sample(pred, gt):
    mse = ((pred - gt) ** 2).mean(dim=[1, 2, 3])
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))


def normalize_bt(bt, vmin, vmax):
    bt = np.clip(bt, vmin, vmax)
    return (vmax - bt) / (vmax - vmin)

def make_composite(bt13, bt8):
    r = normalize_bt(bt13, BT13_MIN, BT13_MAX)
    g = normalize_bt(bt8, BT8_MIN, BT8_MAX)
    b = (r + g) / 2
    rgb = np.dstack((r, g, b))
    return np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)


class GOESRIFEDatasetV4(Dataset):
    def __init__(self, root_seq, root_logs, split="train", augment=True, list_file=None):
        self.root_seq = root_seq
        self.split = split
        self.augment = augment
        with open(os.path.join(root_logs, "manifest.csv"), newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if list_file is None:
            list_file = "tri_trainlist.txt" if split == "train" else "tri_testlist.txt"
        with open(os.path.join(root_logs, list_file)) as f:
            ids = {l.strip() for l in f if l.strip()}
        rows = [r for r in rows if r["seq_id"] in ids]

        self.samples = []
        for r in rows:
            d = os.path.join(root_seq, "sequences", r["seq_id"])
            npz_path = os.path.join(d, "raw_bt.npz")
            if os.path.exists(npz_path):
                self.samples.append({
                    "seq_dir": d, "npz_path": npz_path,
                    "seq_id": r["seq_id"],
                    "half_gap_min": float(r["half_gap_min"])
                })
        print(f"[{split.upper()}] {len(self.samples)} valid samples (from {list_file})")
        gc_ = defaultdict(int)
        for s in self.samples:
            gc_[int(s["half_gap_min"])] += 1
        print(f"[{split.upper()}] Gap dist: {dict(sorted(gc_.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        npz = np.load(s["npz_path"])
        bt13_l, bt8_l = npz["bt13_left"].astype(np.float32), npz["bt8_left"].astype(np.float32)
        bt13_m, bt8_m = npz["bt13_mid"].astype(np.float32),  npz["bt8_mid"].astype(np.float32)
        bt13_r, bt8_r = npz["bt13_right"].astype(np.float32), npz["bt8_right"].astype(np.float32)
        npz.close()

        composite0   = make_composite(bt13_l, bt8_l)
        gt_composite = make_composite(bt13_m, bt8_m)
        composite1   = make_composite(bt13_r, bt8_r)

        tir0 = normalize_bt(bt13_l, BT13_MIN, BT13_MAX)[..., None]
        tir1 = normalize_bt(bt13_r, BT13_MIN, BT13_MAX)[..., None]
        wv0  = normalize_bt(bt8_l,  BT8_MIN,  BT8_MAX)[..., None]
        wv1  = normalize_bt(bt8_r,  BT8_MIN,  BT8_MAX)[..., None]

        gt_tir = normalize_bt(bt13_m, BT13_MIN, BT13_MAX)[..., None]
        gt_wv  = normalize_bt(bt8_m,  BT8_MIN,  BT8_MAX)[..., None]

        def to_chw(arr):
            return torch.from_numpy(np.nan_to_num(arr, nan=0.0).astype(np.float32)).permute(2, 0, 1)

        composite0   = to_chw(composite0)
        composite1   = to_chw(composite1)
        gt_composite = to_chw(gt_composite)
        tir0 = to_chw(tir0); tir1 = to_chw(tir1)
        wv0  = to_chw(wv0);  wv1  = to_chw(wv1)
        gt_tir = to_chw(gt_tir); gt_wv = to_chw(gt_wv)

        gap = torch.tensor(s["half_gap_min"] / 60.0, dtype=torch.float32)

        if self.augment and self.split == "train":
            tensors = [composite0, composite1, gt_composite, gt_tir, gt_wv,
                       tir0, tir1, wv0, wv1]
            if random.random() < 0.5:
                tensors = [torch.flip(t, [2]) for t in tensors]
            if random.random() < 0.5:
                tensors = [torch.flip(t, [1]) for t in tensors]
            if random.random() < 0.5:
                k = random.randint(1, 3)
                tensors = [torch.rot90(t, k, [1, 2]) for t in tensors]
            composite0, composite1, gt_composite, gt_tir, gt_wv, tir0, tir1, wv0, wv1 = tensors

        return composite0, composite1, gt_composite, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap, s["seq_id"]


def filter_by_gap(ds, gap_list):
    if gap_list is None:
        return ds
    if isinstance(ds, Subset):
        base, base_indices = ds.dataset, ds.indices
        keep = [i for i in base_indices if int(base.samples[i]["half_gap_min"]) in gap_list]
    else:
        base, keep = ds, [i for i in range(len(ds)) if int(ds.samples[i]["half_gap_min"]) in gap_list]
    print(f"[TRAIN_GAP_FILTER] Keeping half_gap_min in {gap_list}: "
          f"{len(keep)}/{len(ds)} samples")
    return Subset(base, keep)


def pearson_correlation(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x**2).sum()) * torch.sqrt((y**2).sum()) + 1e-8
    return (x * y).sum() / denom


@torch.no_grad()
def high_freq_ratio(img):
    lap_v = torch.abs(img[:, :, 2:, :] - 2 * img[:, :, 1:-1, :] + img[:, :, :-2, :]).mean(dim=[1, 2, 3])
    lap_h = torch.abs(img[:, :, :, 2:] - 2 * img[:, :, :, 1:-1] + img[:, :, :, :-2]).mean(dim=[1, 2, 3])
    return lap_v + lap_h


def evaluate_mode(model, loader, mode="modified"):
    model.eval()
    metrics = defaultdict(float)
    gap_metrics = defaultdict(lambda: defaultdict(list))
    count = 0

    for batch in tqdm(loader, desc=f"Eval [{mode}]"):
        c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap, _ = batch
        c0 = c0.to(device); c1 = c1.to(device)
        gt_rgb = gt_rgb.to(device); gt_tir = gt_tir.to(device); gt_wv = gt_wv.to(device)
        tir0 = tir0.to(device); tir1 = tir1.to(device)
        wv0 = wv0.to(device); wv1 = wv1.to(device)
        gap = gap.to(device)

        with autocast():
            if mode == "stock":
                final_rgb, base_merged, _, flow, mask, base_tir, base_wv = model.forward(
                    c0, c1, tir0, tir1, wv0, wv1, gap, use_base_only=True)
                pred_rgb = base_merged
                pred_tir = base_tir
                pred_wv  = base_wv
            else:
                final_rgb, base_merged, _, flow, mask, base_tir, base_wv = model.forward(
                    c0, c1, tir0, tir1, wv0, wv1, gap)
                pred_rgb = final_rgb
                pred_tir = final_rgb[:, 0:1, :, :]
                pred_wv  = final_rgb[:, 1:2, :, :]

        psnr_rgb_batch = calculate_psnr_per_sample(pred_rgb.float(), gt_rgb.float())
        psnr_tir_batch = calculate_psnr_per_sample(pred_tir.float(), gt_tir.float())
        psnr_wv_batch  = calculate_psnr_per_sample(pred_wv.float(),  gt_wv.float())
        ssim_batch = 1.0 - ssim_loss(pred_rgb.float(), gt_rgb.float()).item()
        hf_batch = high_freq_ratio(pred_rgb.float())
        gt_hf_batch = high_freq_ratio(gt_rgb.float())

        bs = c0.shape[0]
        count += bs
        metrics["psnr_rgb"] += psnr_rgb_batch.sum().item()
        metrics["psnr_tir"] += psnr_tir_batch.sum().item()
        metrics["psnr_wv"]  += psnr_wv_batch.sum().item()
        metrics["ssim"] += ssim_batch * bs
        metrics["hf"] += hf_batch.sum().item()
        metrics["gt_hf"] += gt_hf_batch.sum().item()

        for i, g in enumerate(gap.cpu().tolist()):
            gk = round(g * 60)
            gap_metrics[gk]["psnr_rgb"].append(psnr_rgb_batch[i].item())
            gap_metrics[gk]["psnr_tir"].append(psnr_tir_batch[i].item())
            gap_metrics[gk]["psnr_wv"].append(psnr_wv_batch[i].item())
            gap_metrics[gk]["hf"].append(hf_batch[i].item())

        del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
        del pred_rgb, pred_tir, pred_wv, final_rgb
        aggressive_gc()

    result = {
        "psnr_rgb": metrics["psnr_rgb"] / count,
        "psnr_tir": metrics["psnr_tir"] / count,
        "psnr_wv":  metrics["psnr_wv"]  / count,
        "ssim":     metrics["ssim"] / count,
        "hf":       metrics["hf"] / count,
        "gt_hf":    metrics["gt_hf"] / count,
        "gap_psnr": {},
    }
    for gk in sorted(gap_metrics.keys()):
        result["gap_psnr"][gk] = {
            "rgb": np.mean(gap_metrics[gk]["psnr_rgb"]),
            "tir": np.mean(gap_metrics[gk]["psnr_tir"]),
            "wv":  np.mean(gap_metrics[gk]["psnr_wv"]),
            "hf":  np.mean(gap_metrics[gk]["hf"]),
            "n":   len(gap_metrics[gk]["psnr_rgb"]),
        }
    return result


@torch.no_grad()
def analyze_correction_direction(model, loader, max_batches=None):
    model.eval()
    stats = defaultdict(list)

    for bi, batch in enumerate(tqdm(loader, desc="Correction direction")):
        if max_batches is not None and bi >= max_batches:
            break
        c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap, _ = batch
        c0 = c0.to(device); c1 = c1.to(device)
        gt_rgb = gt_rgb.to(device); gt_tir = gt_tir.to(device); gt_wv = gt_wv.to(device)
        tir0 = tir0.to(device); tir1 = tir1.to(device)
        wv0 = wv0.to(device); wv1 = wv1.to(device)
        gap = gap.to(device)

        with autocast():
            _, base_merged, _, _, _, _, _ = model.forward(
                c0, c1, tir0, tir1, wv0, wv1, gap, use_base_only=True)
            final_rgb, _, _, _, _, _, _ = model.forward(
                c0, c1, tir0, tir1, wv0, wv1, gap)

        pred_delta = final_rgb - base_merged
        true_delta = gt_rgb - base_merged

        for ch_idx, ch_name in [(0, "tir"), (1, "wv"), (2, "b")]:
            pd = pred_delta[:, ch_idx:ch_idx+1]
            td = true_delta[:, ch_idx:ch_idx+1]
            ss = (torch.sign(pd) == torch.sign(td)).float().mean().item()
            stats[f"corr_sign_{ch_name}"].append(ss)

            flat_idx = torch.randperm(pd.numel(), device=pd.device)[:2000]
            corr = pearson_correlation(pd.flatten()[flat_idx], td.flatten()[flat_idx]).item()
            stats[f"corr_delta_{ch_name}"].append(corr)

        stats["pred_delta_mag"].append(pred_delta.abs().mean().item())
        stats["true_delta_mag"].append(true_delta.abs().mean().item())

        for ch_idx, ch_name in [(0, "tir"), (1, "wv"), (2, "b")]:
            pred_ch = final_rgb[:, ch_idx:ch_idx+1]
            gt_ch = gt_rgb[:, ch_idx:ch_idx+1]
            flat_idx = torch.randperm(pred_ch.numel(), device=pred_ch.device)[:2000]
            corr = pearson_correlation(pred_ch.flatten()[flat_idx], gt_ch.flatten()[flat_idx]).item()
            stats[f"direct_corr_{ch_name}"].append(corr)

        del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
        del base_merged, final_rgb, pred_delta, true_delta
        aggressive_gc()

    return {k: np.mean(v) for k, v in stats.items()}


def check_weight_health(model):
    health = {}
    std = model.refine.head_rgb.weight.detach().std().item()
    health["head_rgb_std"] = std

    other_stds = []
    for n, p in model.refine.named_parameters():
        if "head_rgb" not in n and "weight" in n and p.dim() > 1:
            other_stds.append(p.detach().std().item())
    other_mean = np.mean(other_stds) if other_stds else 1e-8
    health["head_rgb_ratio"] = std / other_mean
    return health


def print_diagnostic_report(stock, mod, dir_stats, health):
    print("" + "╔" + "═"*68 + "╗")
    print("║" + " COMPREHENSIVE DIAGNOSTIC REPORT — v5.4 RANDOM INIT".center(68) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ MODE: STOCK RIFE (frozen base, no refinement)".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")
    print(f"║  RGB PSNR: {stock['psnr_rgb']:.2f} dB  |  TIR PSNR: {stock['psnr_tir']:.2f} dB  |  WV PSNR: {stock['psnr_wv']:.2f} dB".ljust(69) + "║")
    print(f"║  SSIM: {stock['ssim']:.4f}".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ MODE: MODIFIED (v5.4 random init + turbo)".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")
    print(f"║  RGB PSNR: {mod['psnr_rgb']:.2f} dB  |  TIR PSNR: {mod['psnr_tir']:.2f} dB  |  WV PSNR: {mod['psnr_wv']:.2f} dB".ljust(69) + "║")
    print(f"║  SSIM: {mod['ssim']:.4f}".ljust(69) + "║")
    d_rgb = mod["psnr_rgb"] - stock["psnr_rgb"]
    d_tir = mod["psnr_tir"] - stock["psnr_tir"]
    d_wv  = mod["psnr_wv"]  - stock["psnr_wv"]
    print(f"║  Δ vs Stock: RGB={d_rgb:+.3f} dB  TIR={d_tir:+.3f} dB  WV={d_wv:+.3f} dB".ljust(69) + "║")
    print(f"║  Sharpness (Laplacian energy) — Stock: {stock['hf']:.4f}  Mod: {mod['hf']:.4f}  GT: {mod['gt_hf']:.4f}".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ CORRECTION DIRECTION ANALYSIS (pred vs base_merged → gt)".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")
    for ch_name in ["tir", "wv", "b"]:
        ss = dir_stats[f"corr_sign_{ch_name}"]
        cd = dir_stats[f"corr_delta_{ch_name}"]
        dc = dir_stats[f"direct_corr_{ch_name}"]
        flag = "✓ GENUINE" if ss > 0.55 else "⚠ COIN-FLIP" if ss > 0.48 else "🔴 WRONG"
        print(f"║  {ch_name.upper():3s}  same_sign: {ss*100:.1f}%  delta_corr: {cd:+.3f}  direct_corr: {dc:+.3f}  {flag}".ljust(69) + "║")
    rt = dir_stats["pred_delta_mag"] / max(dir_stats["true_delta_mag"], 1e-8)
    print(f"║  |Δ_pred|/|Δ_true|: {rt:.2f}x".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ PER-GAP BREAKDOWN (RGB PSNR + sharpness)".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")
    print("║  Gap   │  Stock    │ Modified  │   Δ     │ Stock hf│ Mod hf  │ Verdict".ljust(69) + "║")
    print("║  ──────┼───────────┼───────────┼─────────┼─────────┼─────────┼─────────".ljust(69) + "║")
    small_gap_max_abs_delta = 0.0
    large_gap_deltas = []
    for gk in sorted(stock["gap_psnr"].keys()):
        s_rgb = stock["gap_psnr"][gk]["rgb"]
        m_rgb = mod["gap_psnr"][gk]["rgb"]
        s_hf = stock["gap_psnr"][gk]["hf"]
        m_hf = mod["gap_psnr"][gk]["hf"]
        dg = m_rgb - s_rgb
        is_small_gap = gk <= GATE_GAP_LOW_MIN
        if is_small_gap:
            small_gap_max_abs_delta = max(small_gap_max_abs_delta, abs(dg))
            v = "✓ SAFE (gated=0)" if abs(dg) < 0.02 else "🔴 GATE LEAK!"
        else:
            large_gap_deltas.append(dg)
            if dg > 0.3:
                v = "✓✓ strong"
            elif dg > -0.05:
                v = "✓ ok (~stock)"
            elif dg > -0.2:
                v = "⚠ mild harm"
            else:
                v = "🔴 harm"
        tag = "[small]" if is_small_gap else "[large]"
        print(f"║  {gk:3d}min │ {s_rgb:7.2f} dB│ {m_rgb:7.2f} dB│ {dg:+6.2f}  │ {s_hf:7.4f} │ {m_hf:7.4f} │ {tag} {v}".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ WEIGHT HEALTH".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")
    std = health["head_rgb_std"]
    ratio = health["head_rgb_ratio"]
    status = "active" if ratio > 0.3 else "small" if ratio > 0.05 else "STUCK"
    print(f"║  head_rgb  std={std:.5f}  ratio={ratio:.3f}x  [{status}]".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    print("║ DIAGNOSTIC VERDICT (small-gap safety / large-gap PSNR / sharpness)".ljust(69) + "║")
    print("╠" + "═"*68 + "╣")

    safety_ok = small_gap_max_abs_delta < 0.02
    print(f"║  1. Small-gap safety: max|Δ| at <={GATE_GAP_LOW_MIN:.0f}min = {small_gap_max_abs_delta:.4f}dB  "
          f"{'✓ PASS' if safety_ok else '🔴 FAIL — gate is leaking'}".ljust(69) + "║")

    if large_gap_deltas:
        mean_large_delta = float(np.mean(large_gap_deltas))
        psnr_ok = mean_large_delta > -0.3
        print(f"║  2. Large-gap (>={GATE_GAP_HIGH_MIN:.0f}min) mean Δ PSNR: {mean_large_delta:+.3f}dB  "
              f"{'✓ OK (at/above stock-level bar)' if psnr_ok else '🔴 WORSE than acceptable'}".ljust(69) + "║")
    else:
        psnr_ok = False
        print("║  2. Large-gap PSNR: no large-gap samples in this eval batch".ljust(69) + "║")

    large_gap_hf = [mod["gap_psnr"][gk]["hf"] for gk in mod["gap_psnr"] if gk >= GATE_GAP_HIGH_MIN]
    large_gap_stock_hf = [stock["gap_psnr"][gk]["hf"] for gk in stock["gap_psnr"] if gk >= GATE_GAP_HIGH_MIN]
    if large_gap_hf:
        mod_hf_large = float(np.mean(large_gap_hf))
        stock_hf_large = float(np.mean(large_gap_stock_hf))
        sharper = mod_hf_large > stock_hf_large
        print(f"║  3. Sharpness @ large gap — Stock hf: {stock_hf_large:.4f}  Mod hf: {mod_hf_large:.4f}  "
              f"{'✓ SHARPER than stock' if sharper else '⚠ not sharper yet'}".ljust(69) + "║")
    else:
        sharper = False

    print("╠" + "═"*68 + "╣")
    if safety_ok and psnr_ok and sharper:
        verdict = "✅ ON TRACK — safe at small gaps, holding/improving PSNR + getting sharper at large gaps."
    elif not safety_ok:
        verdict = "🔴 GATE BUG — small-gap outputs are changing when they structurally shouldn't. Fix before trusting anything else."
    elif not psnr_ok:
        verdict = "🔴 LARGE-GAP PSNR REGRESSING — sharpness push may be too aggressive (try lowering W_GRADIENT)."
    else:
        verdict = "🟡 PARTIAL — safe and PSNR-acceptable, but not visibly sharper yet. Needs more epochs or higher W_GRADIENT."
    print("║  " + verdict.ljust(66) + "║")
    print("╚" + "═"*68 + "╝")


@torch.no_grad()
def run_comprehensive_diagnostic(model, val_loader):
    print("" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC MODE")
    print("Comparing Stock RIFE vs v5.4 Random-Init Modified on IDENTICAL val split")
    print("="*70)

    stock = evaluate_mode(model, val_loader, mode="stock")
    mod   = evaluate_mode(model, val_loader, mode="modified")
    dir_stats = analyze_correction_direction(model, val_loader)
    health = check_weight_health(model)

    print_diagnostic_report(stock, mod, dir_stats, health)
    return stock, mod, dir_stats, health


def get_epoch_config(epoch):
    if epoch < BOOTSTRAP_EPOCHS:
        return BOOTSTRAP_LR, True
    p = (epoch - BOOTSTRAP_EPOCHS) / (NUM_EPOCHS - BOOTSTRAP_EPOCHS)
    lr = REFINE_LR * 0.5 * (1.0 + np.cos(np.pi * p))
    return lr, False


def train_one_epoch(model, loader, optimizer, scaler, epoch, sane_loss_ceiling):
    model.train()
    metrics = defaultdict(float)
    count = 0
    psnr_sum = 0.0
    gap_cnts = defaultdict(int)
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")
    bi = 0

    running_loss_ema = None
    spike_skips = 0
    max_grad_norm_seen = 0.0
    mid_epoch_rollbacks = 0
    consecutive_extreme_grad = 0

    for bi, batch in enumerate(pbar):
        c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap, seq_ids = batch
        c0 = c0.to(device, non_blocking=True); c1 = c1.to(device, non_blocking=True)
        gt_rgb = gt_rgb.to(device, non_blocking=True)
        gt_tir = gt_tir.to(device, non_blocking=True); gt_wv = gt_wv.to(device, non_blocking=True)
        tir0 = tir0.to(device, non_blocking=True); tir1 = tir1.to(device, non_blocking=True)
        wv0 = wv0.to(device, non_blocking=True); wv1 = wv1.to(device, non_blocking=True)
        gap = gap.to(device, non_blocking=True)
        for g in gap.cpu().tolist():
            gap_cnts[round(g * 60)] += 1

        with autocast():
            final_rgb, base_merged, _, _, _, _, _ = model.forward(
                c0, c1, tir0, tir1, wv0, wv1, gap)
            loss, ld = compute_losses(final_rgb, gt_rgb)

        loss_val = loss.item()

        if sane_loss_ceiling is not None and np.isfinite(loss_val) and loss_val > sane_loss_ceiling:
            mid_epoch_rollbacks += 1
            print(f"  [MID-EPOCH] bi={bi} loss={loss_val:.4f} > ceiling {sane_loss_ceiling:.4f}. Rolling back.")
            model.refine.load_state_dict(model.ema_refine.state_dict())
            optimizer.state = defaultdict(dict)
            optimizer.zero_grad()
            running_loss_ema = None
            del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
            del final_rgb, base_merged, loss
            aggressive_gc()
            if mid_epoch_rollbacks >= MAX_MID_EPOCH_ROLLBACKS:
                print(f"  Stopping epoch early after {mid_epoch_rollbacks} rollbacks.")
                break
            continue

        is_spike = (running_loss_ema is not None and bi >= SPIKE_WARMUP_STEPS and
                    loss_val > SPIKE_LOSS_MULTIPLIER * running_loss_ema)
        if is_spike:
            spike_skips += 1
            print(f"  [SPIKE] bi={bi} loss={loss_val:.4f} vs avg={running_loss_ema:.4f} - SKIP")
            del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
            del final_rgb, base_merged, loss
            continue

        running_loss_ema = loss_val if running_loss_ema is None else (
            SPIKE_EMA_DECAY * running_loss_ema + (1 - SPIKE_EMA_DECAY) * loss_val)

        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
        if (bi + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.refine.parameters(), GRAD_CLIP)
            gn = float(grad_norm)
            max_grad_norm_seen = max(max_grad_norm_seen, gn)
            if gn > GRAD_CLIP * 20:
                consecutive_extreme_grad += 1
            else:
                consecutive_extreme_grad = 0
            if consecutive_extreme_grad >= MAX_CONSECUTIVE_EXTREME_GRAD:
                print(f"  [MID-EPOCH] {consecutive_extreme_grad} extreme grads. Rolling back.")
                model.refine.load_state_dict(model.ema_refine.state_dict())
                optimizer.state = defaultdict(dict)
                optimizer.zero_grad()
                scaler.update()
                running_loss_ema = None
                consecutive_extreme_grad = 0
                mid_epoch_rollbacks += 1
                del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
                del final_rgb, base_merged, loss
                aggressive_gc()
                if mid_epoch_rollbacks >= MAX_MID_EPOCH_ROLLBACKS:
                    print("  Stopping epoch early.")
                    break
                continue
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()
            model.update_ema()

        bs = c0.shape[0]; count += bs
        for k, v in ld.items():
            metrics[k] += v * bs
        with torch.no_grad():
            psnr_sum += calculate_psnr(final_rgb.float(), gt_rgb.float()) * bs
            pred_delta_mag = (final_rgb - base_merged).abs().mean().item()
            metrics["pred_delta_mag"] += pred_delta_mag * bs

        del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
        del final_rgb, base_merged, loss
        if bi % 10 == 0:
            aggressive_gc()
            pbar.set_postfix({
                "loss": f"{metrics['total']/count:.4f}" if count else "n/a",
                "psnr": f"{psnr_sum/count:.2f}" if count else "n/a",
                "spikes": spike_skips,
                "rollbacks": mid_epoch_rollbacks})

    if count and (bi + 1) % GRAD_ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.refine.parameters(), GRAD_CLIP)
        max_grad_norm_seen = max(max_grad_norm_seen, float(grad_norm))
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad()
        model.update_ema()

    aggressive_gc()
    gap_str = ", ".join(f"{g}m:{c}" for g,c in sorted(gap_cnts.items()))
    print(f"  [Train gap dist] {gap_str}")
    if spike_skips:
        print(f"  [SPIKE] {spike_skips} microbatch(es) skipped")
    if mid_epoch_rollbacks:
        print(f"  [MID-EPOCH] {mid_epoch_rollbacks} rollback(s)")
    print(f"  [Grad] max pre-clip grad norm: {max_grad_norm_seen:.2f} (clip: {GRAD_CLIP})")
    out = {k: v / count for k, v in metrics.items()} if count else {"total": float("nan")}
    out["psnr"] = psnr_sum / count if count else float("nan")
    out["spike_skips"] = spike_skips
    out["mid_epoch_rollbacks"] = mid_epoch_rollbacks
    out["max_grad_norm"] = max_grad_norm_seen
    return out


@torch.no_grad()
def validate(model, loader, use_ema=False):
    model.eval()
    if use_ema:
        live_state = copy.deepcopy(model.refine.state_dict())
        model.refine.load_state_dict(model.ema_refine.state_dict())

    tp = ts = tl = 0.0
    count = 0
    gap_psnr = defaultdict(list); gap_cnts = defaultdict(int)

    for batch in tqdm(loader, desc="Val"):
        c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap, _ = batch
        c0 = c0.to(device); c1 = c1.to(device)
        gt_rgb = gt_rgb.to(device); gt_tir = gt_tir.to(device); gt_wv = gt_wv.to(device)
        tir0 = tir0.to(device); tir1 = tir1.to(device)
        wv0 = wv0.to(device); wv1 = wv1.to(device); gap = gap.to(device)

        with autocast():
            final_rgb, base_merged, _, flow, mask, _, _ = model.forward(
                c0, c1, tir0, tir1, wv0, wv1, gap)
            loss, _ = compute_losses(final_rgb, gt_rgb)

        pred = final_rgb.float()
        psnr = calculate_psnr(pred, gt_rgb.float())
        ssim_val = 1.0 - ssim_loss(pred, gt_rgb.float()).item()
        bs = c0.shape[0]; count += bs
        tp += psnr * bs; ts += ssim_val * bs; tl += loss.item() * bs
        for g in gap.cpu().tolist():
            gk = round(g * 60); gap_psnr[gk].append(psnr); gap_cnts[gk] += 1
        del c0, c1, gt_rgb, gt_tir, gt_wv, tir0, tir1, wv0, wv1, gap
        del final_rgb, base_merged, flow, mask, pred, loss
        aggressive_gc()

    if use_ema:
        model.refine.load_state_dict(live_state)
    aggressive_gc()
    return {
        "psnr": tp / count, "ssim": ts / count, "loss": tl / count,
        "gap_psnr": {g: np.mean(v) for g, v in sorted(gap_psnr.items())},
        "gap_counts": dict(gap_cnts)
    }


def save_ckpt(model, optimizer, scaler, epoch, best_psnr, path, prev_epoch_loss=None):
    torch.save({
        "epoch": epoch,
        "best_psnr": float(best_psnr),
        "prev_epoch_loss": float(prev_epoch_loss) if prev_epoch_loss is not None else None,
        "refine": model.refine.state_dict(),
        "ema_refine": model.ema_refine.state_dict(),
        "base_flownet": model.base_flownet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)
    print(f"  Saved: {os.path.basename(path)}")

def load_ckpt(model, optimizer, scaler, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.refine.load_state_dict(ckpt["refine"], strict=True)
    if "ema_refine" in ckpt:
        model.ema_refine.load_state_dict(ckpt["ema_refine"], strict=True)
    if not FREEZE_BASE_FLOWNET and "base_flownet" in ckpt:
        model.base_flownet.load_state_dict(ckpt["base_flownet"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    ep, bp = ckpt["epoch"], ckpt.get("best_psnr", 0.0)
    prev_loss = ckpt.get("prev_epoch_loss", None)
    print(f"  Resumed epoch {ep}, best_psnr={bp:.2f}")
    del ckpt; aggressive_gc()
    return ep, bp, prev_loss

def find_latest_ckpt(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None, -1
    best_e, best_p = -1, None
    for f in os.listdir(ckpt_dir):
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
            try:
                n = int(f.split("_")[2].split(".")[0])
                if n > best_e:
                    best_e, best_p = n, os.path.join(ckpt_dir, f)
            except Exception:
                pass
    return best_p, best_e

def cleanup_ckpts(ckpt_dir, keep=KEEP_LAST_N):
    if not os.path.exists(ckpt_dir):
        return
    ckpts = sorted([(int(f.split("_")[2].split(".")[0]), os.path.join(ckpt_dir, f))
                     for f in os.listdir(ckpt_dir)
                     if f.startswith("checkpoint_epoch_") and f.endswith(".pth")])
    for _, p in ckpts[:-keep]:
        try:
            os.remove(p)
            ef = p.replace("checkpoint_epoch_", "epoch_").replace(".pth", "")
            if os.path.exists(ef):
                shutil.rmtree(ef)
        except Exception:
            pass


def run_diagnostic_training(model, optimizer, scaler, train_ds, val_ds):
    global GRAD_ACCUM_STEPS, BATCH_SIZE
    GRAD_ACCUM_STEPS = DIAG_GRAD_ACCUM
    BATCH_SIZE = DIAG_BATCH_SIZE

    train_ds = filter_by_gap(train_ds, TRAIN_GAP_FILTER)

    n_train = max(1, int(len(train_ds) * DIAG_SUBSET_FRACTION))
    n_val = max(1, int(len(val_ds) * DIAG_SUBSET_FRACTION))
    train_idx = random.sample(range(len(train_ds)), n_train)
    val_idx = random.sample(range(len(val_ds)), n_val)
    diag_train_ds = Subset(train_ds, train_idx)
    diag_val_ds = Subset(val_ds, val_idx)

    train_loader = DataLoader(diag_train_ds, batch_size=DIAG_BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(diag_val_ds, batch_size=DIAG_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    print("=" * 70)
    print(f"DIAGNOSTIC MODE - {DIAG_SUBSET_FRACTION*100:.0f}% subset, {DIAG_EPOCHS} epochs")
    print(f"Train: {n_train} samples | Val: {n_val} samples | Eff batch: {DIAG_BATCH_SIZE*DIAG_GRAD_ACCUM}")
    print("=" * 70)

    prev_epoch_loss = None
    sane_loss_ceiling = None
    history = []

    for epoch in range(DIAG_EPOCHS):
        lr, is_bootstrap = get_epoch_config(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        boot_flag = " [BOOTSTRAP]" if is_bootstrap else ""
        print("")
        print("="*60)
        print(f"Diag Epoch {epoch+1}/{DIAG_EPOCHS} | LR: {lr:.2e}{boot_flag}")
        print("="*60)

        tm = train_one_epoch(model, train_loader, optimizer, scaler, epoch, sane_loss_ceiling)
        print(f"Train | total:{tm['total']:.4f} charb_rgb:{tm['charb_rgb']:.4f} charb_tir:{tm['charb_tir']:.4f} "
              f"charb_wv:{tm['charb_wv']:.4f} | PSNR:{tm['psnr']:.2f}dB")

        if prev_epoch_loss is not None and np.isfinite(tm["total"]) and \
                tm["total"] > EPOCH_DIVERGENCE_RATIO * prev_epoch_loss:
            print("  DIVERGENCE - rolling back refine -> EMA + resetting optimizer.")
            model.refine.load_state_dict(model.ema_refine.state_dict())
            optimizer.state = defaultdict(dict)
        elif np.isfinite(tm["total"]):
            prev_epoch_loss = tm["total"]
            sane_loss_ceiling = SANE_LOSS_CEILING_MULTIPLIER * prev_epoch_loss

        stock, mod, dir_stats, health = run_comprehensive_diagnostic(model, val_loader)

        d_rgb = mod["psnr_rgb"] - stock["psnr_rgb"]
        history.append({
            "epoch": epoch + 1,
            "vs_stock_rgb": d_rgb,
            "corr_sign_tir": dir_stats["corr_sign_tir"],
            "corr_sign_wv": dir_stats["corr_sign_wv"],
            "head_rgb_ratio": health["head_rgb_ratio"],
        })
        aggressive_gc()

    print("")
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY TABLE")
    print("=" * 70)
    print("Epoch | vs_stock | cs_tir | cs_wv  | head_rgb")
    for h in history:
        print(f"  {h['epoch']:2d}  | {h['vs_stock_rgb']:+.3f} dB | {h['corr_sign_tir']*100:5.1f}% | "
              f"{h['corr_sign_wv']*100:5.1f}% | {h['head_rgb_ratio']:.4f}")

    if len(history) >= 2:
        last = history[-1]
        trend = last["vs_stock_rgb"]
        if trend > 0.1 and last["corr_sign_tir"] > 0.55 and last["corr_sign_wv"] > 0.55:
            verdict = "READY FOR FULL TRAINING — direct prediction is learning genuine corrections."
        elif trend > 0.02:
            verdict = "PROMISING — trending up but needs more epochs/capacity for full benefit."
        elif trend < -0.05:
            verdict = "DO NOT TRAIN — regression vs stock. Debug before full run."
        else:
            verdict = "AMBIGUOUS — no clear signal. Try longer bootstrap or higher LR."
        print(f"  >>> {verdict}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagnostic", action="store_true",
                    help="Run diagnostic subset training only.")
    ap.add_argument("--cleaned-train-list", default=None,
                    help="Custom train list file inside finetune-logs")
    ap.add_argument("--cleaned-val-list", default=None,
                    help="Custom val list file inside finetune-logs")
    args = ap.parse_args()

    if args.diagnostic:
        global DIAGNOSTIC_MODE
        DIAGNOSTIC_MODE = True

    print("=" * 70)
    print("RIFE Evolution v5.4 — RANDOM INIT + TURBO")
    print("=" * 70)
    print(f"Device: {device} | Batch: {BATCH_SIZE} | Accum: {GRAD_ACCUM_STEPS}")
    print(f"Bootstrap: {BOOTSTRAP_EPOCHS} epochs @ LR={BOOTSTRAP_LR}")
    print(f"FREEZE_BASE_FLOWNET = {FREEZE_BASE_FLOWNET}")
    log_ram("startup")

    for p, n in [(DATA_ROOT_SEQ, "finetune"), (DATA_ROOT_LOGS, "finetune-logs"), (PRETRAIN_PATH, "pretrain")]:
        if not os.path.exists(p):
            print(f"ERROR: Not found: {p} ({n})"); return

    model = Model()
    print(f"Loading pretrained base from {PRETRAIN_PATH} ...")
    model.load_pretrained_base(PRETRAIN_PATH)

    model._refine_forward_compiled = model.refine

    log_ram("after model load")

    param_groups = [
        {"params": model.refine.parameters(), "lr": BOOTSTRAP_LR, "weight_decay": REFINE_WEIGHT_DECAY},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, ADAMW_BETA2))
    scaler = GradScaler()
    start_epoch = 0
    best_psnr = 0.0
    prev_epoch_loss = None
    sane_loss_ceiling = None

    rp, _ = find_latest_ckpt(CHECKPOINT_DIR)
    if rp:
        print(f"Auto-resume: {rp}")
        loaded_ep, best_psnr, loaded_prev_loss = load_ckpt(model, optimizer, scaler, rp)
        start_epoch = loaded_ep + 1
        if loaded_prev_loss is not None:
            prev_epoch_loss = loaded_prev_loss
            sane_loss_ceiling = SANE_LOSS_CEILING_MULTIPLIER * prev_epoch_loss
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Fresh training start")

    train_ds = GOESRIFEDatasetV4(DATA_ROOT_SEQ, DATA_ROOT_LOGS, "train", augment=True,
                                  list_file=args.cleaned_train_list)
    val_ds   = GOESRIFEDatasetV4(DATA_ROOT_SEQ, DATA_ROOT_LOGS, "val", augment=False,
                                  list_file=args.cleaned_val_list)
    if len(train_ds) == 0:
        raise RuntimeError("No training samples! Check raw_bt.npz exists.")
    if len(val_ds) == 0:
        print("No val samples - using 10% of train")
        n_val = max(1, int(len(train_ds) * 0.1))
        idx = list(range(len(train_ds))); random.shuffle(idx)
        val_base = GOESRIFEDatasetV4(DATA_ROOT_SEQ, DATA_ROOT_LOGS, "train", augment=False,
                                      list_file=args.cleaned_train_list)
        val_ds = Subset(val_base, idx[:n_val]); train_ds = Subset(train_ds, idx[n_val:])

    if DIAGNOSTIC_MODE:
        run_diagnostic_training(model, optimizer, scaler, train_ds, val_ds)
        return

    train_ds = filter_by_gap(train_ds, TRAIN_GAP_FILTER)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    print(f"  Training: epoch {start_epoch+1} -> {NUM_EPOCHS}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    epochs_since_improve = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        lr, is_bootstrap = get_epoch_config(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        boot_flag = " [BOOTSTRAP]" if is_bootstrap else ""
        print("")
        print("="*60)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {lr:.2e} | Best: {best_psnr:.2f} dB{boot_flag}")
        print("="*60)

        tm = train_one_epoch(model, train_loader, optimizer, scaler, epoch, sane_loss_ceiling)
        print(f"Train | total:{tm['total']:.4f} charb_rgb:{tm['charb_rgb']:.4f} "
              f"charb_tir:{tm['charb_tir']:.4f} charb_wv:{tm['charb_wv']:.4f} "
              f"| PSNR:{tm['psnr']:.2f}dB")
        if tm.get("mid_epoch_rollbacks", 0):
            print(f"  [MID-EPOCH] {tm['mid_epoch_rollbacks']} rollback(s) this epoch")

        if prev_epoch_loss is not None and np.isfinite(tm["total"]) and \
                tm["total"] > EPOCH_DIVERGENCE_RATIO * prev_epoch_loss:
            print(f"  DIVERGENCE: epoch loss {tm['total']:.4f} > {EPOCH_DIVERGENCE_RATIO}x {prev_epoch_loss:.4f}")
            model.refine.load_state_dict(model.ema_refine.state_dict())
            optimizer.state = defaultdict(dict)
        elif np.isfinite(tm["total"]):
            prev_epoch_loss = tm["total"]
            sane_loss_ceiling = SANE_LOSS_CEILING_MULTIPLIER * prev_epoch_loss

        if (epoch + 1) % VAL_EVERY == 0:
            vm = validate(model, val_loader, use_ema=True)
            print(f"Val (EMA) | PSNR:{vm['psnr']:.2f}dB SSIM:{vm['ssim']:.4f} Loss:{vm['loss']:.4f}")
            for g, p in vm["gap_psnr"].items():
                print(f"  {g:>3}min: {p:.2f}dB")

            gap_db = tm["psnr"] - vm["psnr"]
            print(f"  [Overfitting] train-val PSNR gap: {gap_db:+.2f} dB")
            if gap_db > TRAIN_VAL_GAP_WARN_DB:
                print("  OVERFITTING WARNING")

            health = check_weight_health(model)
            print(f"  [Health] head_rgb_ratio={health['head_rgb_ratio']:.3f}")

            if vm["psnr"] > best_psnr:
                best_psnr = vm["psnr"]
                epochs_since_improve = 0
                bp = os.path.join(CHECKPOINT_DIR, "best_model")
                os.makedirs(bp, exist_ok=True)
                model.save_refine(bp, use_ema=True)
                save_ckpt(model, optimizer, scaler, epoch, best_psnr,
                          os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth"),
                          prev_epoch_loss=prev_epoch_loss)
                print(f"*** NEW BEST: {best_psnr:.2f} dB ***")
            else:
                epochs_since_improve += 1
                if EARLY_STOP_ENABLED and epochs_since_improve >= EARLY_STOP_PATIENCE:
                    print(f"  EARLY STOP: no improvement for {EARLY_STOP_PATIENCE} validations.")
                    break

        ep_dir = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_refine(ep_dir, use_ema=False)
        save_ckpt(model, optimizer, scaler, epoch, best_psnr,
                  os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth"),
                  prev_epoch_loss=prev_epoch_loss)
        cleanup_ckpts(CHECKPOINT_DIR)
        aggressive_gc()

    fp = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(fp, exist_ok=True)
    model.save_refine(fp, use_ema=True)
    save_ckpt(model, optimizer, scaler, NUM_EPOCHS - 1, best_psnr,
              os.path.join(CHECKPOINT_DIR, "final_checkpoint.pth"),
              prev_epoch_loss=prev_epoch_loss)
    print("")
    print("=" * 70)
    print(f"DONE | Best Val PSNR (EMA): {best_psnr:.2f} dB")
    print(f"Best refine: {CHECKPOINT_DIR}/best_model/refine_state.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()

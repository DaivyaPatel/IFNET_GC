import os
import sys
import re
import tempfile
import urllib.request
import torch
import cv2
import numpy as np
import xarray as xr

from datetime import datetime

# ------------------------------------------------------------------
# Import your model architecture (unchanged)
# ------------------------------------------------------------------
from app.ml.pysteps_run import run_pysteps

from app.ml.evolution_model import (
    Model,
    make_rgb,
    normalize_bt,
    BT13_MIN, BT13_MAX,
    BT8_MIN, BT8_MAX,
    pad_to_multiple,
    unpad
)

import pathlib

try:
    from pysteps import motion
    from pysteps.extrapolation.semilagrangian import extrapolate
    PYSTEPS_AVAILABLE = True
except ImportError:
    PYSTEPS_AVAILABLE = False



import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_PATH = os.getenv(
    "RIFE_GC_CHECKPOINT_PATH",
    os.path.join(str(ROOT_DIR), "checkpoints", "best_checkpoint.pth")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


# ==============================================================
#  UNIVERSAL DATA LOADING (ported from universal_6thaugust.py)
# ==============================================================

# For INSAT .h5 files: which image datasets to use for TIR / WV.
INSAT_TIR_DATASET = "IMG_TIR1"
INSAT_WV_DATASET  = "IMG_WV"

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def detect_format(filepath):
    """Infer file format from extension."""
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
    base = os.path.basename(filepath)
    fmt = detect_format(filepath)

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


def load_nc(path):
    """
    Loads a GOES .nc CMI band file. The CMI variable is ALREADY
    calibrated brightness temperature in Kelvin. DQF is used as a
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

    The main image datasets (IMG_TIR1 / IMG_WV etc.) are RAW
    10-bit instrument counts (uint16). The temperature is obtained
    via a per-file calibration LOOKUP TABLE shipped in the same .h5
    (IMG_TIR1_TEMP / IMG_WV_TEMP, each length 1024, indexed by raw
    count). Pixels equal to the dataset's _FillValue (1023) are
    masked as invalid.
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


# ==============================================================
#  MODEL LOADING (unchanged)
# ==============================================================

def get_model():
    global _model
    if _model is None:
        print("=" * 50)
        print("Loading RIFE Evolution Model...")
        print(f"Checkpoint: {CHECKPOINT_PATH}")
        print("=" * 50)

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            )

        _model = Model()
        _model.load_checkpoint(CHECKPOINT_PATH, use_ema=True)
        _model.eval()
        
        print(f"  Model ready on {device}")
        print("=" * 50)
    return _model


# ==============================================================
#  HELPERS
# ==============================================================

def download_to_tempfile(url: str, suffix: str = ".nc") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


# ==============================================================
#  INTERPOLATION API (unchanged signature)
# ==============================================================

def interpolate(
    img0_url: str,
    img1_url: str,
    tir0_url: str,
    tir1_url: str,
    wv0_url: str,
    wv1_url: str,
    gap_minutes: float = 15.0,
    model_type: str = "ifnet-gc",
) -> bytes:
    """
    Interpolates a midpoint frame between two satellite observations.

    Parameters
    ----------
    img0_url, img1_url : str
        Reserved for future use (RGB inputs). Currently the RGB is
        derived from the TIR/WV bands below.
    tir0_url, tir1_url : str
        URLs to the TIR (CH13 / IMG_TIR1) files for frame 0 and 1.
    wv0_url, wv1_url : str
        URLs to the WV (CH08 / IMG_WV) files for frame 0 and 1.
        For INSAT .h5, this can be the SAME URL as tir0/tir1 since
        both bands live in one file.
    gap_minutes : float
        Temporal gap between the two frames (for logging only).

    Returns
    -------
    bytes
        PNG-encoded interpolated RGB frame.
    """
    if model_type == "ifnet-gc":
        model = get_model()

    print("Downloading Raw Images...")
    tir0_path = download_to_tempfile(tir0_url)
    tir1_path = download_to_tempfile(tir1_url)
    wv0_path = download_to_tempfile(wv0_url)
    wv1_path = download_to_tempfile(wv1_url)
    print("Raw Images Downloaded")

    try:
        # ------------------------------------------------------------------
        # Universal band loading: auto-detects GOES .nc vs INSAT .h5
        # ------------------------------------------------------------------
        bt13_0, bt8_0 = load_band_pair(tir0_path, wv0_path)
        bt13_1, bt8_1 = load_band_pair(tir1_path, wv1_path)

        rgb_0 = make_rgb(bt13_0, bt8_0)
        rgb_1 = make_rgb(bt13_1, bt8_1)

        def to_t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

        def band_to_t(bt, vmin, vmax):
            n = normalize_bt(bt, vmin, vmax)
            n = np.nan_to_num(n, nan=0.0).astype(np.float32)
            return torch.from_numpy(n).unsqueeze(0).unsqueeze(0).float().to(device)

        img0 = to_t(rgb_0)
        img1 = to_t(rgb_1)
        tir0_t = band_to_t(bt13_0, BT13_MIN, BT13_MAX)
        tir1_t = band_to_t(bt13_1, BT13_MIN, BT13_MAX)
        wv0_t  = band_to_t(bt8_0,  BT8_MIN,  BT8_MAX)
        wv1_t  = band_to_t(bt8_1,  BT8_MIN,  BT8_MAX)

        # Pad to multiple of 32
        img0, orig_hw = pad_to_multiple(img0, 32)
        img1, _       = pad_to_multiple(img1, 32)
        tir0_t, _     = pad_to_multiple(tir0_t, 32)
        tir1_t, _     = pad_to_multiple(tir1_t, 32)
        wv0_t, _      = pad_to_multiple(wv0_t, 32)
        wv1_t, _      = pad_to_multiple(wv1_t, 32)

        print(f"Running inference with {model_type} (gap={gap_minutes:.2f} min)...")

        with torch.no_grad():
            if model_type == "ifnet-gc":
                final, base_merged, residual = model.inference(
                    img0, img1, tir0_t, tir1_t, wv0_t, wv1_t, scale=1.0)
                final = unpad(final, orig_hw)
                pred = np.clip(final[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)
            elif model_type == "linear":
                tir_interp = (bt13_0 + bt13_1) / 2.0
                wv_interp = (bt8_0 + bt8_1) / 2.0
                pred = make_rgb(tir_interp, wv_interp)
            elif model_type == "pysteps":
                pred = run_pysteps(bt13_0, bt8_0, bt13_1, bt8_1)[0]
            elif model_type == "rife":
                from app.ml.train_log.RIFE_HDv3 import Model as RifeModel
                rife_model = RifeModel(local_rank=-1)
                rife_model.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_log"), rank=-1)
                rife_model.eval()
                final = rife_model.inference(img0, img1, scale=1.0)
                final = unpad(final, orig_hw)
                pred = np.clip(final[0].permute(1, 2, 0).cpu().numpy(), 0, 1).astype(np.float32)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        result_np = (pred * 255.0).clip(0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(".png", result_bgr)
        if not success:
            raise RuntimeError("Failed to encode interpolated frame as PNG")

        print("Inference complete, returning PNG bytes")
        return encoded.tobytes()

    finally:
        # Cleanup temp files
        for p in [tir0_path, tir1_path, wv0_path, wv1_path]:
            try:
                os.remove(p)
            except OSError:
                pass
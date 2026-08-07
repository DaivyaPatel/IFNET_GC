import io
import re
from datetime import datetime, timezone

import h5py
import numpy as np
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

# Confirmed variable names per satellite/format/channel combination
GOES_VAR_NAME = "CMI"          # same variable name for both TIR and WV in GOES CMIP single-band files
INSAT_TIR_VAR_NAME = "IMG_TIR1"
INSAT_WV_VAR_NAME = "IMG_WV"

# Per-channel Kelvin normalization ranges, matching the reference RGB
# composite logic exactly. TIR and WV use DIFFERENT ranges.
BT13_MIN, BT13_MAX = 190.0, 310.0   # TIR range
BT8_MIN, BT8_MAX = 190.0, 280.0     # WV range

CROP_SIZE = 256


# ── METADATA EXTRACTION (capture time from file, not upload time) ──

def _parse_iso_datetime(dt_str: str) -> datetime:
    """Parse ISO 8601 datetime string to timezone-aware datetime."""
    dt_str = dt_str.strip()
    # Handle Z suffix
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    # Handle fractional seconds with more than 6 digits (Python limitation)
    if "." in dt_str:
        before_dot, after_dot = dt_str.split(".", 1)
        # Keep only up to 6 digits of fractional seconds
        frac = re.sub(r"[^0-9]", "", after_dot)[:6]
        tz_match = re.search(r"[+-]\d{2}:?\d{2}$", after_dot)
        if tz_match:
            dt_str = f"{before_dot}.{frac}{tz_match.group()}"
        else:
            dt_str = f"{before_dot}.{frac}"
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # Fallback: try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
            try:
                return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {dt_str}")


def _extract_goes_timestamp_from_filename(filename: str) -> datetime:
    """Fallback: parse GOES timestamp from filename (YYYYDDDHHMMSSf format)."""
    m = re.search(r"_s(\d{14})\d?_", filename)
    if not m:
        m = re.search(r"_s(\d{13})_", filename)
    if not m:
        raise ValueError(f"No GOES timestamp found in filename: {filename}")
    ts = m.group(1)
    year = int(ts[0:4])
    doy = int(ts[4:7])
    hh = int(ts[7:9])
    mm = int(ts[9:11])
    ss = int(ts[11:13])
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + __import__("datetime").timedelta(
        days=doy - 1, hours=hh, minutes=mm, seconds=ss
    )
    return dt


def _extract_insat_timestamp_from_filename(filename: str) -> datetime:
    """Fallback: parse INSAT timestamp from filename (e.g., MG_23JUN2026_0000_...)."""
    # Pattern: DDMMMYYYY_HHMM
    m = re.search(r"(\d{2})([A-Z]{3})(\d{4})[_-](\d{4})", filename.upper())
    if m:
        day, month_str, year, time_str = m.groups()
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        month = month_map.get(month_str, 1)
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        return datetime(int(year), month, int(day), hh, mm, 0, tzinfo=timezone.utc)
    raise ValueError(f"No INSAT timestamp found in filename: {filename}")


def extract_capture_time_nc(file_bytes: bytes, filename: str) -> datetime:
    """
    Extract capture time from GOES NetCDF (.nc) file metadata.
    Tries global attributes in order: time_coverage_start, time_coverage_end, date_created.
    Falls back to filename parsing if metadata is missing.
    """
    try:
        ds = xr.open_dataset(io.BytesIO(file_bytes))
        attrs = ds.attrs

        for attr_name in ("time_coverage_start", "time_coverage_end", "date_created", "start_time"):
            if attr_name in attrs and attrs[attr_name]:
                try:
                    dt = _parse_iso_datetime(str(attrs[attr_name]))
                    ds.close()
                    return dt
                except Exception:
                    continue

        # Try the 'time' coordinate if it exists
        if "time" in ds.coords and len(ds.coords["time"]) > 0:
            try:
                t_val = ds.coords["time"].values[0]
                if hasattr(t_val, "isoformat"):
                    dt = _parse_iso_datetime(t_val.isoformat())
                    ds.close()
                    return dt
            except Exception:
                pass

        ds.close()
    except Exception:
        pass

    # Fallback to filename
    return _extract_goes_timestamp_from_filename(filename)


def extract_capture_time_h5(file_bytes: bytes, filename: str) -> datetime:
    """
    Extract capture time from INSAT HDF5 (.h5) file metadata.
    Tries root attributes in order: NominalTime, ObservationTime, Date+Time.
    Falls back to filename parsing if metadata is missing.
    """
    try:
        f = h5py.File(io.BytesIO(file_bytes), "r")
        attrs = dict(f.attrs)

        for attr_name in ("NominalTime", "ObservationTime", "ImageTime", "SceneTime"):
            if attr_name in attrs and attrs[attr_name]:
                try:
                    val = attrs[attr_name]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    dt = _parse_iso_datetime(str(val))
                    f.close()
                    return dt
                except Exception:
                    continue

        # Try Date + Time as separate attributes
        if "Date" in attrs and "Time" in attrs:
            try:
                date_val = attrs["Date"]
                time_val = attrs["Time"]
                if isinstance(date_val, bytes):
                    date_val = date_val.decode("utf-8")
                if isinstance(time_val, bytes):
                    time_val = time_val.decode("utf-8")
                dt_str = f"{str(date_val).strip()} {str(time_val).strip()}"
                dt = _parse_iso_datetime(dt_str)
                f.close()
                return dt
            except Exception:
                pass

        f.close()
    except Exception:
        pass

    # Fallback to filename
    return _extract_insat_timestamp_from_filename(filename)


def extract_capture_time(file_bytes: bytes, filename: str) -> datetime:
    """
    Extract capture time from satellite file metadata.
    Supports GOES (.nc) and INSAT (.h5) formats.
    Returns timezone-aware datetime (UTC).
    """
    lower_name = filename.lower()
    if lower_name.endswith(".nc"):
        return extract_capture_time_nc(file_bytes, filename)
    elif lower_name.endswith(".h5") or lower_name.endswith(".hdf5"):
        return extract_capture_time_h5(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type for metadata extraction: {filename}")


def compute_gap_minutes(
    t0_bytes: bytes,
    t0_filename: str,
    t1_bytes: bytes,
    t1_filename: str,
) -> float:

    dt0 = extract_capture_time(t0_bytes, t0_filename)
    dt1 = extract_capture_time(t1_bytes, t1_filename)
    if dt1 < dt0:
        raise ValueError(
        "The second image must be captured after the first image."
    )

    return abs((dt1 - dt0).total_seconds()) / 60.0

# ── RGB COMPOSITE BUILDING (unchanged, matches testing script exactly) ──

def _extract_channel_array(file_bytes: bytes, filename: str, channel_type: str) -> np.ndarray:
    """
    Extracts the raw brightness-temperature array from a GOES (.nc) or
    INSAT (.h5) file, based on file extension and the explicitly-declared
    channel_type ("tir" or "wv").
    """
    lower_name = filename.lower()

    if lower_name.endswith(".nc"):
        ds = xr.open_dataset(io.BytesIO(file_bytes))
        arr = ds[GOES_VAR_NAME].values
        ds.close()

    elif lower_name.endswith(".h5") or lower_name.endswith(".hdf5"):
        f = h5py.File(io.BytesIO(file_bytes), "r")
        available_keys = list(f.keys())
        
        dataset_name = None
        if channel_type == "tir":
            for cand in ["TIR1_BT", "IMG_TIR1", "IMG_TIR2", "TIR2_BT", "CMI"]:
                if cand in available_keys:
                    dataset_name = cand
                    break
        elif channel_type == "wv":
            for cand in ["WV_BT", "IMG_WV", "IMG_MIR", "CMI"]:
                if cand in available_keys:
                    dataset_name = cand
                    break
                    
        if not dataset_name:
            for k in available_keys:
                if k not in ["X", "Y", "lat", "lon", "Latitude", "Longitude"]:
                    dataset_name = k
                    break
                    
        if not dataset_name:
            raise KeyError(f"Could not find a valid {channel_type} dataset in {filename}. Available: {available_keys}")
            
        raw_ds = f[dataset_name]
        raw = raw_ds[:]
        raw = np.squeeze(raw)
        
        fill_value = None
        if "_FillValue" in raw_ds.attrs:
            fv = raw_ds.attrs["_FillValue"]
            fill_value = int(np.asarray(fv).flatten()[0])
            
        lut_name = f"{dataset_name}_TEMP"
        if lut_name in f:
            # Requires LUT conversion (raw instrument counts)
            raw = raw.astype(np.int32)
            lut = f[lut_name][:].astype(np.float32)
            idx = np.clip(raw, 0, lut.shape[0] - 1)
            arr = lut[idx]
        else:
            # Already calibrated brightness temperature (e.g. TIR1_BT)
            arr = raw.astype(np.float32)
            
        if fill_value is not None:
            arr = np.where(raw == fill_value, np.nan, arr)
            
        f.close()

    else:
        raise ValueError(f"Unsupported file type: {filename}")

    # arr is already squeezed for INSAT above, but squeeze for GOES just in case
    arr = np.squeeze(arr)
    return arr.astype(np.float32)


def normalize_bt(bt: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Normalizes a Kelvin brightness-temperature array to [0, 1].
    INVERTED on purpose (colder = brighter):
        (vmax - bt) / (vmax - vmin)
    """
    clipped = np.clip(bt, vmin, vmax)
    normalized = (vmax - clipped) / (vmax - vmin)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized


def _center_crop(arr: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    h, w = arr.shape
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image too small to crop to {crop_size}x{crop_size}: got {h}x{w}"
        )
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    return arr[start_y:start_y + crop_size, start_x:start_x + crop_size]


def preprocess_channel(file_bytes: bytes, filename: str, channel_type: str) -> np.ndarray:
    """
    Extracts and crops the raw Kelvin brightness-temperature array for a
    single uploaded file. Returns a (256, 256) float32 array of raw Kelvin values.
    """
    arr = _extract_channel_array(file_bytes, filename, channel_type)
    cropped = _center_crop(arr)
    return cropped


def build_rgb_composite(tir_bytes: bytes, tir_filename: str, wv_bytes: bytes, wv_filename: str) -> bytes:
    """
    Takes raw TIR and WV file bytes and combines them into an RGB composite PNG:
        R = normalize_bt(TIR, BT13_MIN, BT13_MAX)
        G = normalize_bt(WV,  BT8_MIN,  BT8_MAX)
        B = average(R, G)
    Returns PNG-encoded bytes.
    """
    tir_raw = preprocess_channel(tir_bytes, tir_filename, "tir")
    wv_raw = preprocess_channel(wv_bytes, wv_filename, "wv")

    r = normalize_bt(tir_raw, BT13_MIN, BT13_MAX)
    b = normalize_bt(wv_raw, BT8_MIN, BT8_MAX)
    g = (r + b) / 2.0

    rgb = np.dstack((r, g, b))
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 1).astype(np.float32)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ── GAP MAP (visualizes the scalar time-gap value as an image) ──

def build_gap_map(
    gap_minutes: float,
    size: int = CROP_SIZE,
    background_color: tuple = (20, 20, 30),
    text_color: tuple = (255, 255, 255),
) -> bytes:
    """
    Renders the scalar time-gap (in minutes) as a simple labeled square PNG,
    so it can be stored/displayed alongside the RGB composites.

    Returns PNG-encoded bytes.
    """
    if gap_minutes is None:
        raise ValueError("gap_minutes is None; cannot build gap map.")

    img = Image.new("RGB", (size, size), color=background_color)
    draw = ImageDraw.Draw(img)

    gap_hours = gap_minutes / 60.0
    title_text = "TIME GAP"
    value_text = f"{gap_minutes:.2f} min"
    subtext = f"({gap_hours:.4f} hrs)"

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size // 16)
        value_font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size // 10)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", size=size // 18)
    except Exception:
        title_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    def _centered(text, font, y):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = (size - text_w) / 2
        draw.text((x, y), text, fill=text_color, font=font)

    _centered(title_text, title_font, size * 0.30)
    _centered(value_text, value_font, size * 0.42)
    _centered(subtext, sub_font, size * 0.60)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
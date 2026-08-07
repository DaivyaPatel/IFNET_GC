import os
import io
import re
import warnings
import numpy as np
import h5py
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

warnings.filterwarnings("ignore")

COLORMAP_DICT = {
    "TIR": "hot",
    "WV": "cool",
    "VIS": "gray",
    "Rad": "viridis",
    "DEFAULT": "viridis"
}

EXCLUDE_DATASETS = {
    "Latitude", "Longitude", "DQF", "time", "x", "y",
    "goes_imager_projection", "yaw_flip", "scan_line_attributes",
    "processing_parm", "quality_factors", "solar_zenith_angle",
    "solar_azimuth_angle", "sensor_zenith_angle", "sensor_azimuth_angle",
    "t", "t_star_look", "band_wavelength_star_look", "star_id",
    "channel_integration_time", "channel_gain_field"
}

def get_timestamp_from_filename(filename):
    base = os.path.basename(filename)
    m = re.search(r"_s(\d{14})_", base)
    if m:
        return m.group(1)
    m = re.search(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", base)
    if m:
        date_str, hhmm = m.groups()
        try:
            dt = datetime.strptime(date_str + hhmm, "%d%b%Y%H%M")
        except ValueError:
            return None
        doy = dt.timetuple().tm_yday
        return f"{dt.year:04d}{doy:03d}{dt.hour:02d}{dt.minute:02d}00"
    return None

def get_goes_band_name(filename):
    base = os.path.basename(filename)
    m = re.search(r"C(\d{2})", base)
    return f"C{m.group(1)}" if m else "unknown"

def choose_colormap(dataset_name, variable_name=""):
    name_upper = dataset_name.upper()
    if "RAD" in variable_name.upper() or "RAD" in name_upper:
        return COLORMAP_DICT["Rad"]
    for key, cmap in COLORMAP_DICT.items():
        if key in name_upper:
            return cmap
    return COLORMAP_DICT["DEFAULT"]

def normalize_image(data, vmin=None, vmax=None, percentiles=(2, 98)):
    data = data.astype(np.float32)
    if vmin is None or vmax is None:
        p_low, p_high = percentiles
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return np.zeros_like(data)
        vmin = np.nanpercentile(valid, p_low)
        vmax = np.nanpercentile(valid, p_high)
    if vmax == vmin:
        vmax = vmin + 1e-6
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0, 1)

def read_goes_nc(filepath):
    ds = xr.open_dataset(filepath)
    var_name = None
    if "CMI" in ds:
        var_name = "CMI"
        units = "K"
        is_rad = False
    elif "Rad" in ds:
        var_name = "Rad"
        units = "W/m^2 sr μm"
        is_rad = True
    else:
        ds.close()
        raise KeyError("No 'CMI' or 'Rad' variable found in this .nc file.")
    
    data = ds[var_name].values.astype(np.float32)
    if "DQF" in ds:
        dqf = ds["DQF"].values
        data = np.where(dqf <= 1, data, np.nan)
    ds.close()

    band = get_goes_band_name(filepath)
    if not is_rad:
        if band.startswith("C13") or band.startswith("C14"):
            vmin, vmax = 190, 310
        elif band.startswith("C08") or band.startswith("C09"):
            vmin, vmax = 190, 280
        elif band.startswith("C02"):
            vmin, vmax = 0, 1.0
        else:
            vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    else:
        vmin, vmax = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
    
    return {
        "data": data,
        "band_name": band,
        "vmin": vmin,
        "vmax": vmax,
        "units": units,
        "is_rad": is_rad
    }

def is_image_dataset(dataset):
    shape = dataset.shape
    if len(shape) == 2:
        return shape[0] > 10 and shape[1] > 10
    elif len(shape) == 3:
        return shape[0] == 1 and shape[1] > 10 and shape[2] > 10
    return False

def read_insat_h5(filepath):
    with h5py.File(filepath, "r") as f:
        dataset_names = []
        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_names.append(name)
        f.visititems(collect_datasets)

        image_datasets = []
        for name in dataset_names:
            if any(excl in name for excl in EXCLUDE_DATASETS):
                continue
            obj = f[name]
            if not is_image_dataset(obj):
                continue
            image_datasets.append(name)

        for chan_name in image_datasets:
            ds = f[chan_name]
            raw = ds[:]
            if len(raw.shape) == 3 and raw.shape[0] == 1:
                raw = np.squeeze(raw, axis=0)
            raw = raw.astype(np.float32)

            fill_value = ds.attrs.get("_FillValue", None)
            if fill_value is not None:
                if hasattr(fill_value, "shape") and fill_value.shape == ():
                    fill_value = fill_value.item()
                if isinstance(fill_value, bytes):
                    fill_value = float(fill_value)
                raw = np.where(raw == fill_value, np.nan, raw)

            lut_name = f"{chan_name}_TEMP"
            if lut_name in f:
                lut = f[lut_name][:].astype(np.float32)
                idx = np.clip(raw, 0, lut.shape[0] - 1).astype(np.int32)
                calibrated = lut[idx]
                calibrated = np.where(np.isnan(raw), np.nan, calibrated)
                data = calibrated
                units = "K"
                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
                else:
                    vmin, vmax = 190, 310
            else:
                data = raw
                units = "counts"
                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
                else:
                    vmin, vmax = 0, 1

            yield {
                "data": data,
                "channel_name": chan_name,
                "vmin": vmin,
                "vmax": vmax,
                "units": units
            }

def process_file_in_memory(filepath, original_filename):
    ext = os.path.splitext(original_filename)[1].lower()
    ts = get_timestamp_from_filename(original_filename)
    if ts is None:
        ts = "unknown"
        
    results = []
    
    if ext == ".nc":
        info = read_goes_nc(filepath)
        data = info["data"]
        band = info["band_name"]
        vmin, vmax = info["vmin"], info["vmax"]
        is_rad = info.get("is_rad", False)

        norm = normalize_image(data, vmin, vmax)
        norm = np.nan_to_num(norm, nan=0.0)

        cmap_name = choose_colormap(band, variable_name="Rad" if is_rad else "CMI")
        cmap = plt.get_cmap(cmap_name)
        colored = cmap(norm)[:, :, :3]

        img_byte_arr = io.BytesIO()
        Image.fromarray((colored * 255).astype(np.uint8)).save(img_byte_arr, format='PNG')
        
        results.append({
            "name": f"{band}_{ts}.png",
            "bytes": img_byte_arr.getvalue()
        })

    elif ext in (".h5", ".hdf5"):
        for chan_info in read_insat_h5(filepath):
            data = chan_info["data"]
            chan_name = chan_info["channel_name"]
            vmin, vmax = chan_info["vmin"], chan_info["vmax"]

            norm = normalize_image(data, vmin, vmax)
            norm = np.nan_to_num(norm, nan=0.0)

            cmap_name = choose_colormap(chan_name)
            cmap = plt.get_cmap(cmap_name)
            colored = cmap(norm)[:, :, :3]

            safe_chan = re.sub(r"[^a-zA-Z0-9]", "_", chan_name)
            img_byte_arr = io.BytesIO()
            Image.fromarray((colored * 255).astype(np.uint8)).save(img_byte_arr, format='PNG')
            
            results.append({
                "name": f"{safe_chan}_{ts}.png",
                "bytes": img_byte_arr.getvalue()
            })
            
    return results

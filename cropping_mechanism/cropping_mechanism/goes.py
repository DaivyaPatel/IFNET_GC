#!/usr/bin/env python3
"""
goes.py — GOES-16/17/18/19 ABI L1b/L2 NetCDF4 (.nc) support.

Everything GOES-specific lives here. Unlike the original script, this
reads .nc files with netCDF4 (a real NetCDF reader) instead of h5py
duck-typing on internal key names. That means:
  - Classic-format NetCDF3 files are opened correctly (netCDF4 handles
    both transparently) instead of silently failing an h5py-only check.
  - A GOES-like file with unexpected internal structure raises a clear
    error instead of returning `kind = None` and vanishing from output.
"""

import os
import numpy as np

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    NCDataset = None

from geo_utils import nearest_index, compute_window, decode

FORMAT_NAME = "goes"

_KNOWN_CHANNEL_VARS = ("Rad", "CMI", "DQF", "BCM", "Power", "Phase")


class GoesFormatError(Exception):
    """Raised when a file looks like GOES but can't actually be parsed
    (missing projection info, unexpected grid vars, etc). Deliberately
    not swallowed — core.py surfaces it instead of dropping the file."""
    pass


def can_open(path):
    """
    Cheap, format-only check: does this look like a GOES ABI file?
    Opens with netCDF4 (works for both classic and HDF5-backed NetCDF4),
    not h5py, so format quirks that confused the old h5py-only sniff no
    longer cause a silent misclassification.
    """
    if not path.lower().endswith(".nc"):
        return False
    if NCDataset is None:
        raise RuntimeError("netCDF4 is required to read .nc files: pip install netCDF4")
    try:
        with NCDataset(path, "r") as f:
            return "goes_imager_projection" in f.variables
    except OSError:
        # Not actually a valid NetCDF file at all.
        return False


def list_channels(path):
    with NCDataset(path, "r") as f:
        chans = [k for k in _KNOWN_CHANNEL_VARS if k in f.variables]
        if not chans:
            # Generic fallback: any 2D var matching the (y, x) dims.
            ny = f.dimensions["y"].size
            nx = f.dimensions["x"].size
            for k, var in f.variables.items():
                if var.ndim == 2 and var.shape == (ny, nx):
                    chans.append(k)
        if not chans:
            raise GoesFormatError(
                f"{path}: no recognizable channel variable found "
                f"(looked for {_KNOWN_CHANNEL_VARS} and any 2D (y,x) var)."
            )
        return chans


def _scan_angle_coords(f):
    """Return (x_rad, y_rad) 1-D arrays of true scan angles in radians,
    decoded from the packed integer x/y + scale_factor/add_offset.
    netCDF4 auto-applies scale_factor/add_offset by default, so we read
    the raw values with auto-scaling disabled to be explicit and avoid
    relying on that implicit behavior."""
    x_var = f.variables["x"]
    y_var = f.variables["y"]

    x_var.set_auto_maskandscale(False)
    y_var.set_auto_maskandscale(False)
    x_raw = x_var[:]
    y_raw = y_var[:]

    x_scale = float(np.asarray(getattr(x_var, "scale_factor", 1.0)).ravel()[0])
    x_off = float(np.asarray(getattr(x_var, "add_offset", 0.0)).ravel()[0])
    y_scale = float(np.asarray(getattr(y_var, "scale_factor", 1.0)).ravel()[0])
    y_off = float(np.asarray(getattr(y_var, "add_offset", 0.0)).ravel()[0])

    x_rad = x_raw.astype(np.float64) * x_scale + x_off
    y_rad = y_raw.astype(np.float64) * y_scale + y_off
    return x_rad, y_rad


def _lonlat_to_scan_angles(lon, lat, proj_attrs):
    """
    Invert the GOES-R ABI Fixed Grid geostationary projection:
    geodetic (lon, lat) -> (x, y) scan angles in radians.
    Standard formulas from the GOES-R PUG (Vol 3, Sec 5.1.2.8).
    """
    req = float(np.asarray(proj_attrs["semi_major_axis"]).ravel()[0])
    rpol = float(np.asarray(proj_attrs["semi_minor_axis"]).ravel()[0])
    H = float(np.asarray(proj_attrs["perspective_point_height"]).ravel()[0]) + req
    lon0 = float(np.asarray(proj_attrs["longitude_of_projection_origin"]).ravel()[0])

    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    lon0_r = np.radians(lon0)

    e2 = 1 - (rpol ** 2) / (req ** 2)
    phi_c = np.arctan((rpol ** 2 / req ** 2) * np.tan(lat_r))
    rc = rpol / np.sqrt(1 - e2 * (np.cos(phi_c) ** 2))

    sx = H - rc * np.cos(phi_c) * np.cos(lon_r - lon0_r)
    sy = -rc * np.cos(phi_c) * np.sin(lon_r - lon0_r)
    sz = rc * np.sin(phi_c)

    rl = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    cond = (H * (H - sx)) < (sy ** 2 + (req ** 2 / rpol ** 2) * sz ** 2)
    if np.any(cond):
        raise ValueError(
            "Requested center lat/lon is not visible from this GOES "
            "satellite's viewing geometry (behind the Earth's limb)."
        )

    y_scan = np.arctan(sz / sx)
    x_scan = np.arcsin(-sy / rl)
    return float(x_scan), float(y_scan)


def grid_signature(path, channel):
    """
    Hashable signature identifying the (x,y) scan-angle grid used by
    `channel`. Calls the SAME _scan_angle_coords() used by crop_channel,
    so the signature can never drift out of sync with the actual crop
    (the original script duplicated this logic inline — a source of
    subtle locking bugs).
    """
    with NCDataset(path, "r") as f:
        x_rad, y_rad = _scan_angle_coords(f)
        shp = f.variables[channel].shape
        return (
            "goes",
            round(float(x_rad[0]), 8), round(float(x_rad[-1]), 8), len(x_rad),
            round(float(y_rad[0]), 8), round(float(y_rad[-1]), 8), len(y_rad),
            tuple(shp),
        )


def crop_channel(path, channel, center_lat, center_lon, patch_size,
                  window_override=None):
    with NCDataset(path, "r") as f:
        if "goes_imager_projection" not in f.variables:
            raise GoesFormatError(f"{path}: missing goes_imager_projection variable.")
        proj_attrs = {k: f.variables["goes_imager_projection"].getncattr(k)
                      for k in f.variables["goes_imager_projection"].ncattrs()}
        x_rad, y_rad = _scan_angle_coords(f)

        var = f.variables[channel]
        var.set_auto_maskandscale(False)  # we apply scale/offset ourselves, explicitly
        data = np.asarray(var[:])
        attrs = {k: var.getncattr(k) for k in var.ncattrs()}

        nrows, ncols = data.shape  # y, x

        if window_override is not None:
            row_start, row_end, col_start, col_end = window_override
            ok = (row_end - row_start == patch_size) and \
                 (col_end - col_start == patch_size) and \
                 (0 <= row_start) and (row_end <= nrows) and \
                 (0 <= col_start) and (col_end <= ncols)
        else:
            x_scan, y_scan = _lonlat_to_scan_angles(center_lon, center_lat, proj_attrs)
            col_center = nearest_index(x_rad, x_scan)
            row_center = nearest_index(y_rad, y_scan)
            row_start, row_end, ok_r = compute_window(nrows, row_center, patch_size)
            col_start, col_end, ok_c = compute_window(ncols, col_center, patch_size)
            ok = ok_r and ok_c

        cropped = data[row_start:row_end, col_start:col_end]

        if "scale_factor" in attrs and "add_offset" in attrs:
            scale = float(np.asarray(attrs["scale_factor"]).ravel()[0])
            offset = float(np.asarray(attrs["add_offset"]).ravel()[0])
            fill = attrs.get("_FillValue")
            cropped_phys = cropped.astype(np.float64) * scale + offset
            if fill is not None:
                fill_val = np.asarray(fill).ravel()[0]
                cropped_phys = np.where(cropped == fill_val, np.nan, cropped_phys)
        else:
            cropped_phys = cropped

        band_id = None
        if "band_id" in f.variables:
            band_id = int(np.asarray(f.variables["band_id"][:]).ravel()[0])

        platform = getattr(f, "platform_ID", "GOES")

        return {
            "satellite": decode(platform),
            "channel": f"{channel}" + (f"_C{band_id:02d}" if band_id else ""),
            "array": cropped,
            "array_physical": cropped_phys,
            "row_window": (row_start, row_end),
            "col_window": (col_start, col_end),
            "fits_in_bounds": ok,
            "attrs": {k: decode(v) for k, v in attrs.items()},
            "x_crop": x_rad[col_start:col_end],
            "y_crop": y_rad[row_start:row_end],
            "source_file": os.path.basename(path),
            "kind": FORMAT_NAME,
        }

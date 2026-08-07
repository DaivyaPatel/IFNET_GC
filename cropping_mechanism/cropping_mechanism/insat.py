#!/usr/bin/env python3
"""
insat.py — INSAT-3D/3DR/3DS Imager L1B/L1C HDF5 (.h5) support.

Everything INSAT-specific lives here: format detection, projection math,
channel discovery, and cropping. core.py never touches h5py directly for
INSAT files — it only calls the functions below.
"""

import os
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from pyproj import CRS, Transformer

from geo_utils import nearest_index, compute_window, decode

FORMAT_NAME = "insat"

# Known INSAT Imager channel datasets (L1B/L1C). Used only as a default
# friendly filter — channel discovery itself is dynamic (see list_channels).
KNOWN_CHANNELS = [
    "IMG_VIS", "IMG_SWIR", "IMG_MIR", "IMG_WV", "IMG_TIR1", "IMG_TIR2",
]

_NON_CHANNEL_KEYS = {
    "Projection_Information", "X", "Y", "time", "proj_dim", "GreyCount",
}

_SUPPORTED_PROJECTIONS = {"mercator"}


class InsatFormatError(Exception):
    """Raised when a file looks like INSAT but can't actually be parsed
    (e.g. unsupported projection). This is deliberately NOT swallowed —
    core.py surfaces it instead of silently dropping the file."""
    pass


def can_open(path):
    """
    Cheap, format-only check: does this look like an INSAT L1B/L1C file?
    Does NOT validate that we can actually crop it (projection support
    etc.) — that's checked eagerly in crop_channel / grid_signature and
    raises InsatFormatError if unsupported, rather than pretending the
    file doesn't exist.
    """
    if not path.lower().endswith((".h5", ".hdf5")):
        return False
    if h5py is None:
        raise RuntimeError("h5py is required to read .h5 files: pip install h5py")
    try:
        with h5py.File(path, "r") as f:
            return "Projection_Information" in f or any(
                k.startswith("IMG_") for k in f.keys()
            )
    except OSError:
        # Not actually a valid HDF5 file at all.
        return False


def list_channels(path):
    """Return list of channel dataset names present in an INSAT h5 file."""
    with h5py.File(path, "r") as f:
        chans = []
        for k in f.keys():
            if k in _NON_CHANNEL_KEYS:
                continue
            obj = f[k]
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                chans.append(k)
        return chans


def _transformer_from_proj_attrs(proj_attrs):
    """
    Build a pyproj Transformer (lon,lat -> x,y in meters) matching the
    projection described in Projection_Information attrs of an INSAT
    L1B/L1C HDF5 file.

    Raises InsatFormatError (not silently) if the grid_mapping isn't one
    we know how to invert, so bad/unsupported files show up as errors
    instead of vanishing from the output.
    """
    grid_mapping = proj_attrs.get("grid_mapping_name", b"mercator")
    if isinstance(grid_mapping, (bytes, np.bytes_)):
        grid_mapping = grid_mapping.decode()
    grid_mapping = grid_mapping.lower()

    if grid_mapping not in _SUPPORTED_PROJECTIONS:
        raise InsatFormatError(
            f"INSAT grid_mapping '{grid_mapping}' is not supported. "
            f"Supported: {sorted(_SUPPORTED_PROJECTIONS)}. "
            f"Add its proj4 string in insat.py:_transformer_from_proj_attrs()."
        )

    semi_major = float(np.asarray(proj_attrs["semi_major_axis"]).ravel()[0])
    semi_minor = float(np.asarray(proj_attrs["semi_minor_axis"]).ravel()[0])
    lon_0 = float(np.asarray(proj_attrs["longitude_of_projection_origin"]).ravel()[0])
    false_easting = float(np.asarray(proj_attrs.get("false_easting", [0.0])).ravel()[0])
    false_northing = float(np.asarray(proj_attrs.get("false_northing", [0.0])).ravel()[0])
    lat_ts = float(np.asarray(proj_attrs["standard_parallel"]).ravel()[0])

    proj4 = (
        f"+proj=merc +lon_0={lon_0} +lat_ts={lat_ts} "
        f"+x_0={false_easting} +y_0={false_northing} "
        f"+a={semi_major} +b={semi_minor} +units=m +no_defs"
    )

    crs_geo = CRS.from_epsg(4326)  # WGS84 lat/lon
    crs_proj = CRS.from_proj4(proj4)
    return Transformer.from_crs(crs_geo, crs_proj, always_xy=True)


def grid_signature(path, channel):
    """
    A hashable signature identifying the (X,Y) grid used by `channel`,
    so core.py can detect which datasets/files share an identical grid
    (needed for "locked" alignment mode). Pure metadata read — no crop
    math — so it can never disagree with the actual crop.
    """
    with h5py.File(path, "r") as f:
        x = f["X"]
        y = f["Y"]
        shape = f[channel].shape[-2:]
        return (
            "insat",
            round(float(x[0]), 3), round(float(x[-1]), 3), len(x),
            round(float(y[0]), 3), round(float(y[-1]), 3), len(y),
            tuple(shape),
        )


def crop_channel(path, channel, center_lat, center_lon, patch_size,
                  window_override=None):
    """
    Crop one channel dataset from an INSAT h5 file to a patch_size x
    patch_size square centered on (center_lat, center_lon).

    window_override: optional (row_start, row_end, col_start, col_end)
    to force a specific pixel window (used for "locked" alignment mode
    across channels that share the same grid).
    """
    with h5py.File(path, "r") as f:
        proj_attrs = dict(f["Projection_Information"].attrs)
        x_coords = f["X"][:]        # shape (ncols,)
        y_coords = f["Y"][:]        # shape (nrows,)
        ds = f[channel]
        data = ds[:]                # shape (1, nrows, ncols) typically
        attrs = dict(ds.attrs)

        data2d = data[0] if data.ndim == 3 else data
        nrows, ncols = data2d.shape

        if window_override is not None:
            row_start, row_end, col_start, col_end = window_override
            ok = (row_end - row_start == patch_size) and \
                 (col_end - col_start == patch_size) and \
                 (0 <= row_start) and (row_end <= nrows) and \
                 (0 <= col_start) and (col_end <= ncols)
        else:
            transformer = _transformer_from_proj_attrs(proj_attrs)
            x_m, y_m = transformer.transform(center_lon, center_lat)
            col_center = nearest_index(x_coords, x_m)
            row_center = nearest_index(y_coords, y_m)
            row_start, row_end, ok_r = compute_window(nrows, row_center, patch_size)
            col_start, col_end, ok_c = compute_window(ncols, col_center, patch_size)
            ok = ok_r and ok_c

        cropped = data2d[row_start:row_end, col_start:col_end]

        return {
            "satellite": decode(f.attrs.get("Satellite_Name", b"INSAT")),
            "channel": channel,
            "array": cropped,
            "row_window": (row_start, row_end),
            "col_window": (col_start, col_end),
            "fits_in_bounds": ok,
            "attrs": {k: decode(v) for k, v in attrs.items()
                      if k not in ("DIMENSION_LIST",)},
            "x_crop": x_coords[col_start:col_end],
            "y_crop": y_coords[row_start:row_end],
            "source_file": os.path.basename(path),
            "kind": FORMAT_NAME,
        }

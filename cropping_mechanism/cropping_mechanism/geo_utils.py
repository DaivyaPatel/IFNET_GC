#!/usr/bin/env python3
"""
geo_utils.py — format-agnostic pixel-window math shared by insat.py and
goes.py. Kept intentionally dumb: no file I/O, no format knowledge.
"""

import numpy as np


def nearest_index(arr_1d, value):
    """Index of the element in a monotonic 1D array closest to value."""
    arr_1d = np.asarray(arr_1d)
    idx = int(np.argmin(np.abs(arr_1d - value)))
    return idx


def compute_window(n_total, center_idx, patch_size):
    """
    Compute [start, end) pixel window of length patch_size centered on
    center_idx along an axis of length n_total, clamped to stay in-bounds
    (shifted, not shrunk, when the center is near an edge).

    Returns (start, end, ok) where ok=False means the patch does not fit
    at all within the array (patch_size > n_total) or the center is so
    far outside the array that no shift keeps it in range.
    """
    if patch_size > n_total:
        return 0, n_total, False

    half_lo = patch_size // 2
    half_hi = patch_size - half_lo  # handles odd patch sizes

    start = center_idx - half_lo
    end = center_idx + half_hi

    if start < 0:
        end += -start
        start = 0
    if end > n_total:
        start -= (end - n_total)
        end = n_total

    ok = (start >= 0) and (end <= n_total) and (end - start == patch_size)
    return start, end, ok


def decode(v):
    """Decode bytes/np.bytes_/ndarray-of-bytes attrs into plain python."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "S":
            return [x.decode(errors="replace") for x in v]
        return v.tolist()
    return v

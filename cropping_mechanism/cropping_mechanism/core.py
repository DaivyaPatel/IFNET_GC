#!/usr/bin/env python3
"""
core.py — orchestration for the universal lat/lon square-patch cropper.

Discovers .h5/.nc files, dispatches each to insat.py or goes.py, applies
same-grid "locking" so datasets sharing a grid get bit-identical pixel
windows, writes cropped NetCDF outputs, and writes a manifest.

Design goal (unchanged from the original single-file script): given a
folder of files (any mix of INSAT .h5 and GOES .nc, any channels, any
native resolution) and a target (center_lat, center_lon, patch_size),
every dataset is cropped to the same physical lat/lon square, computed
independently per-dataset from that dataset's own geolocation grid.

Key differences from the old single-file version (see rewrite notes):
  - Format dispatch is explicit and fails loudly. A file that matches
    neither format, or matches a format but can't actually be parsed
    (unsupported projection, missing vars), is recorded in
    manifest["skipped"] with a reason instead of silently vanishing.
  - There is exactly ONE implementation of the locking/window-reuse
    logic (below). The old script had a second, unused, inconsistent
    copy (`get_window_for`) plus an inline duplicate of GOES grid-
    signature computation — both removed.
"""

import os
import glob
import json
import argparse
import numpy as np

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    NCDataset = None

import insat
import goes

# Each format module must expose the same interface:
#   can_open(path) -> bool
#   list_channels(path) -> list[str]
#   grid_signature(path, channel) -> hashable
#   crop_channel(path, channel, lat, lon, size, window_override=None) -> dict
FORMAT_MODULES = [insat, goes]


# --------------------------------------------------------------------------
# Discovery & dispatch
# --------------------------------------------------------------------------

def discover_files(path):
    """
    Accepts either a single .h5/.hdf5/.nc file path or a folder. If given
    a single file, returns just that file (still going through the same
    downstream classify/crop logic as a folder of one). If given a folder,
    recurses and globs as before.
    """
    if os.path.isfile(path):
        if path.lower().endswith((".h5", ".hdf5", ".nc")):
            return [path]
        raise ValueError(
            f"{path} is not a .h5/.hdf5/.nc file (unrecognized extension)."
        )

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = sorted(set(
        glob.glob(os.path.join(path, "**", "*.h5"), recursive=True)
        + glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True)
        + glob.glob(os.path.join(path, "**", "*.nc"), recursive=True)
    ))
    return files


def classify_file(path):
    """
    Return the format module that claims this file, or None with a
    reason string if nothing claims it. Unlike the old script, this
    never returns a bare None with no explanation — callers always get
    (module_or_None, reason_or_None).
    """
    matches = []
    errors = []
    for mod in FORMAT_MODULES:
        try:
            if mod.can_open(path):
                matches.append(mod)
        except Exception as e:
            errors.append(f"{mod.FORMAT_NAME}: {e}")

    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        names = [m.FORMAT_NAME for m in matches]
        return None, f"ambiguous — matched multiple formats: {names}"
    if errors:
        return None, f"no format matched; errors while checking: {'; '.join(errors)}"
    return None, "no format matched (not a recognized INSAT or GOES file)"


# --------------------------------------------------------------------------
# Locking: single implementation shared by both formats
# --------------------------------------------------------------------------

def _build_jobs(files, channels_filter, lock_same_grid, skipped):
    """
    Pass 1: enumerate every (file, channel) job, classify the file, list
    its channels, and (if locking is enabled) compute its grid signature.

    Any failure here (unsupported projection, unreadable channel list,
    etc.) is recorded in `skipped` with a reason rather than raised —
    one bad file shouldn't abort a whole-folder batch run — but it is
    always visible in the manifest, never silent.
    """
    jobs = []
    for path in files:
        mod, reason = classify_file(path)
        if mod is None:
            skipped.append({"path": path, "reason": reason})
            continue

        try:
            chans = mod.list_channels(path)
        except Exception as e:
            skipped.append({"path": path, "reason": f"list_channels failed: {e}"})
            continue

        if channels_filter:
            chans = [c for c in chans if any(
                cf.lower() in c.lower() for cf in channels_filter)]

        for ch in chans:
            sig = None
            if lock_same_grid:
                try:
                    sig = mod.grid_signature(path, ch)
                except Exception as e:
                    # Locking is best-effort: if we can't compute a
                    # signature, this dataset just doesn't get locked
                    # to anything (still cropped independently below).
                    skipped.append({
                        "path": path, "channel": ch,
                        "reason": f"grid_signature failed, will crop unlocked: {e}",
                        "fatal": False,
                    })
            jobs.append({"path": path, "mod": mod, "channel": ch, "grid_sig": sig})
    return jobs


def _crop_with_locking(jobs, center_lat, center_lon, patch_size):
    """
    Pass 2: for each grid signature, compute the pixel window ONCE (from
    the first job encountered with that signature) and reuse it — via
    window_override — for every other job sharing that signature. Jobs
    with no signature (locking disabled or signature failed) are always
    cropped independently.

    This is the single, authoritative implementation of "locked" mode —
    replacing the old script's dead `get_window_for` plus its separate
    inline reuse loop, which could silently disagree with each other.
    """
    window_by_sig = {}   # sig -> (row_start, row_end, col_start, col_end)
    result_by_sig = {}   # sig -> first-computed result dict (avoid recompute)
    results = []

    for job in jobs:
        sig = job["grid_sig"]

        if sig is not None and sig in window_by_sig:
            result = job["mod"].crop_channel(
                job["path"], job["channel"], center_lat, center_lon,
                patch_size, window_override=window_by_sig[sig])
        else:
            result = job["mod"].crop_channel(
                job["path"], job["channel"], center_lat, center_lon, patch_size)
            if sig is not None:
                window_by_sig[sig] = (result["row_window"][0], result["row_window"][1],
                                       result["col_window"][0], result["col_window"][1])
                result_by_sig[sig] = result

        results.append((job, result))

    return results


# --------------------------------------------------------------------------
# Folder-level orchestration
# --------------------------------------------------------------------------

def crop_folder(path, center_lat, center_lon, patch_size, out_dir,
                 channels_filter=None, lock_same_grid=True):
    """
    Main entry point. Accepts either a single .h5/.hdf5/.nc file path or
    a folder containing such files. Crops every recognized file/channel
    to a patch_size x patch_size square at (center_lat, center_lon).

    Returns a manifest dict describing everything written AND everything
    skipped (with reasons) — nothing disappears silently.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = discover_files(path)
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5/.nc files found under {path}")

    manifest = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "patch_size": patch_size,
        "input_path": path,
        "outputs": [],
        "skipped": [],
    }

    jobs = _build_jobs(files, channels_filter, lock_same_grid, manifest["skipped"])
    if not jobs:
        print(f"[WARN] No croppable channels found across {len(files)} file(s). "
              f"See manifest['skipped'] for reasons.")
        manifest_path = os.path.join(out_dir, "crop_manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return manifest

    results = _crop_with_locking(jobs, center_lat, center_lon, patch_size)

    for job, result in results:
        if not result["fits_in_bounds"]:
            print(f"[WARN] {job['path']} [{job['channel']}]: requested patch "
                  f"partially/fully outside available grid, or window_override "
                  f"did not fit — check output shape before using this crop.")

        base = os.path.splitext(os.path.basename(job["path"]))[0]
        out_name = f"{base}__{job['channel']}.nc"
        out_path = os.path.join(out_dir, out_name)

        write_crop_to_nc(out_path, result)

        manifest["outputs"].append({
            "source_file": job["path"],
            "channel": job["channel"],
            "kind": job["mod"].FORMAT_NAME,
            "output": out_path,
            "shape": list(result["array"].shape),
            "row_window": result["row_window"],
            "col_window": result["col_window"],
            "grid_signature_group": job["grid_sig"],
            "fits_in_bounds": result["fits_in_bounds"],
        })

    manifest_path = os.path.join(out_dir, "crop_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    n_skipped_fatal = len([s for s in manifest["skipped"] if s.get("fatal", True)])
    if n_skipped_fatal:
        print(f"[WARN] {n_skipped_fatal} file(s)/channel(s) skipped — see "
              f"manifest['skipped'] in {manifest_path} for reasons.")

    return manifest


# --------------------------------------------------------------------------
# NetCDF output
# --------------------------------------------------------------------------

def _sanitize_attr_value(v):
    """netCDF4 attrs must be str/number/1D-array-of-those, not dicts/None/bool."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (str, int, float)):
        return v
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, (str, int, float)) for x in v):
            return v
        return str(v)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("U", "S", "O"):
            return str(v.tolist())
        return v
    return str(v)


def write_crop_to_nc(out_path, result):
    """
    Write one cropped dataset to a self-describing NetCDF4 (.nc) file.
    Stores the cropped array (and array_physical if present, e.g. GOES
    scaled/offset data) plus the cropped x/y coordinate arrays, the pixel
    window used, and all original dataset attributes.
    """
    if NCDataset is None:
        raise SystemExit("netCDF4 is required to write .nc output: pip install netCDF4")

    array = result["array"]
    nrows, ncols = array.shape

    with NCDataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", nrows)
        ds.createDimension("x", ncols)

        var = ds.createVariable("array", array.dtype, ("y", "x"),
                                 zlib=True, complevel=4)
        var[:, :] = array

        if "array_physical" in result:
            arr_phys = result["array_physical"]
            var_phys = ds.createVariable("array_physical", "f8", ("y", "x"),
                                          zlib=True, complevel=4,
                                          fill_value=np.nan)
            var_phys[:, :] = arr_phys

        x_crop = np.asarray(result["x_crop"])
        y_crop = np.asarray(result["y_crop"])
        x_var = ds.createVariable("x", x_crop.dtype, ("x",))
        x_var[:] = x_crop
        y_var = ds.createVariable("y", y_crop.dtype, ("y",))
        y_var[:] = y_crop

        ds.satellite = str(result.get("satellite", ""))
        ds.channel = str(result.get("channel", ""))
        ds.source_file = str(result.get("source_file", ""))
        ds.row_window_start, ds.row_window_end = result["row_window"]
        ds.col_window_start, ds.col_window_end = result["col_window"]
        ds.fits_in_bounds = int(bool(result["fits_in_bounds"]))

        for k, v in result.get("attrs", {}).items():
            try:
                setattr(ds, f"orig_{k}", _sanitize_attr_value(v))
            except Exception:
                pass  # skip anything netCDF4 truly can't store


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Universal lat/lon square-patch cropper for INSAT-3D/3DR/3DS "
                    "and GOES ABI satellite imagery.")
    ap.add_argument("path", help="A single .h5/.hdf5/.nc file, OR a folder "
                                  "containing such files (searched recursively)")
    ap.add_argument("--lat", type=float, required=True, help="Center latitude")
    ap.add_argument("--lon", type=float, required=True, help="Center longitude")
    ap.add_argument("--size", type=int, required=True,
                     help="Patch size in pixels (square), e.g. 256")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--channels", nargs="*", default=None,
                     help="Optional substring filter on channel names, "
                          "e.g. --channels TIR1 VIS")
    ap.add_argument("--no-lock", action="store_true",
                     help="Disable same-grid alignment locking (independent "
                          "nearest-pixel crop per dataset instead)")
    args = ap.parse_args()

    manifest = crop_folder(
        args.path, args.lat, args.lon, args.size, args.out,
        channels_filter=args.channels,
        lock_same_grid=not args.no_lock,
    )
    print(f"\nDone. {len(manifest['outputs'])} crops written to {args.out}")
    if manifest["skipped"]:
        print(f"{len(manifest['skipped'])} skipped — see crop_manifest.json for reasons")
    print(f"Manifest: {os.path.join(args.out, 'crop_manifest.json')}")


if __name__ == "__main__":
    main()
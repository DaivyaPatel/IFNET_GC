#!/usr/bin/env python3
"""
Dense HSV Motion Vector Visualizer
====================================
Generates high-contrast dense HSV optical flow images with thick,
white-outlined black arrows overlaid. Designed for satellite frame
interpolation dashboards (IFNET-GC / React).

Usage:
    python hsv_motion_viz.py --pred pred.flo --gt gt.flo --out ./viz/
    python hsv_motion_viz.py --pred pred.npy --gt gt.npy --out ./viz/

Backend API:
    from hsv_motion_viz import visualize_flow
    images = visualize_flow(pred_flow, gt_flow, step=20, arrow_scale=3.0)
    # images is a dict of np.ndarray (BGR) ready for cv2.imwrite
"""

import os
import struct
import argparse
from pathlib import Path
from typing import Optional, Dict, Union

import numpy as np
import cv2


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    ARROW_STEP = 20           # Pixel spacing between arrows (lower = denser)
    ARROW_SCALE = 3.0         # Multiplier for arrow length
    ARROW_THICKNESS = 2       # Arrow shaft thickness in px
    MIN_MAG = 1.0             # Skip arrows below this magnitude
    MAX_MAG_PCTILE = 99.5     # Percentile for saturation cap
    WHEEL_SIZE = 180          # Color wheel legend size in px


# =============================================================================
# FLOW I/O
# =============================================================================

def read_flo(path: str) -> np.ndarray:
    """Read Middlebury .flo file -> (H, W, 2) float32."""
    with open(path, "rb") as f:
        tag = struct.unpack("f", f.read(4))[0]
        if tag != 202021.25:
            raise ValueError(f"Bad .flo file: {path}")
        w, h = struct.unpack("ii", f.read(8))
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape((h, w, 2))


def read_flow(path: str) -> np.ndarray:
    """Read .flo or .npy -> (H, W, 2) float32."""
    p = Path(path)
    if p.suffix == ".flo":
        return read_flo(path)
    elif p.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[-1] != 2:
            raise ValueError(f"Flow last dim must be 2, got {arr.shape}")
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported: {p.suffix}. Use .flo or .npy")


# =============================================================================
# CORE VISUALIZATION
# =============================================================================

def flow_to_hsv(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Dense HSV color encoding. Returns BGR uint8."""
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    if max_mag is None or max_mag < 1e-3:
        max_mag = np.percentile(mag, Config.MAX_MAG_PCTILE)
        if max_mag < 1e-3:
            max_mag = 1.0

    h = ang / 2.0                           # 0-180 for OpenCV HSV
    s = np.clip(mag / max_mag, 0, 1) * 255
    val = np.ones_like(mag) * 255

    static = mag < 0.5
    val[static] = 180
    s[static] = 0

    hsv = np.stack([h, s, val], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(img: np.ndarray, flow: np.ndarray,
                step: int = 20, scale: float = 3.0,
                thickness: int = 2, min_mag: float = 1.0) -> np.ndarray:
    """
    Overlay highly visible arrows on image.
    White outline + black body = readable on ANY HSV color.
    """
    out = img.copy()
    h, w = flow.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x] * scale
            mag = np.sqrt(dx**2 + dy**2)
            if mag < min_mag:
                continue

            x2 = int(np.clip(x + dx, 0, w - 1))
            y2 = int(np.clip(y + dy, 0, h - 1))

            # White outline (thicker)
            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(255, 255, 255),
                            thickness=thickness + 2,
                            tipLength=0.35,
                            line_type=cv2.LINE_AA)
            # Black body
            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(0, 0, 0),
                            thickness=thickness,
                            tipLength=0.35,
                            line_type=cv2.LINE_AA)
    return out


def make_color_wheel(size: int = 180) -> np.ndarray:
    """Standard optical flow color wheel legend. Returns RGB uint8."""
    yy, xx = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    mag = np.sqrt(xx**2 + yy**2)
    ang = np.arctan2(-yy, -xx)
    mask = mag <= 1.0

    h = (ang + np.pi) / (2 * np.pi) * 180
    s = np.clip(mag, 0, 1) * 255
    v = np.ones_like(mag) * 255

    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~mask] = 255
    return rgb


# =============================================================================
# MAIN API
# =============================================================================

def visualize_flow(
    pred_flow: Union[str, np.ndarray],
    gt_flow: Optional[Union[str, np.ndarray]] = None,
    step: int = None,
    arrow_scale: float = None,
    max_mag: float = None,
) -> Dict[str, np.ndarray]:
    """
    Generate all motion vector visualizations.

    Args:
        pred_flow: Path (.flo/.npy) or (H,W,2) array.
        gt_flow:   Optional ground-truth flow.
        step:      Arrow subsample step (default Config.ARROW_STEP).
        arrow_scale: Arrow length multiplier (default Config.ARROW_SCALE).
        max_mag:   Shared magnitude cap. Auto-computed if None.

    Returns:
        Dict of BGR uint8 images:
            "pred"          -> HSV + arrows for predicted flow
            "gt"            -> HSV + arrows for ground truth
            "side_by_side"  -> Both labeled, with gap
            "diff"          -> Magnitude error heatmap
            "wheel"         -> RGB color wheel legend
    """
    step = step if step is not None else Config.ARROW_STEP
    arrow_scale = arrow_scale if arrow_scale is not None else Config.ARROW_SCALE

    # Resolve inputs
    def _resolve(x):
        return read_flow(x) if isinstance(x, str) else x.astype(np.float32)

    pred = _resolve(pred_flow)
    gt = _resolve(gt_flow) if gt_flow is not None else None

    # Shared max magnitude
    if max_mag is None:
        mags = []
        for fl in [pred, gt]:
            if fl is not None:
                m, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
                mags.append(m)
        if mags:
            max_mag = np.percentile(np.concatenate([m.ravel() for m in mags]),
                                    Config.MAX_MAG_PCTILE)
        else:
            max_mag = 1.0
    if max_mag < 1e-3:
        max_mag = 1.0

    results: Dict[str, np.ndarray] = {}

    # --- Predicted ---
    hsv_pred = flow_to_hsv(pred, max_mag)
    results["pred"] = draw_arrows(hsv_pred, pred, step=step,
                                   scale=arrow_scale,
                                   thickness=Config.ARROW_THICKNESS,
                                   min_mag=Config.MIN_MAG)

    if gt is not None:
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

        # --- Ground Truth ---
        hsv_gt = flow_to_hsv(gt, max_mag)
        results["gt"] = draw_arrows(hsv_gt, gt, step=step,
                                     scale=arrow_scale,
                                     thickness=Config.ARROW_THICKNESS,
                                     min_mag=Config.MIN_MAG)

        # --- Side by side ---
        gap = np.ones((pred.shape[0], 20, 3), dtype=np.uint8) * 255
        combined = np.hstack([results["pred"], gap, results["gt"]])

        label_h = 40
        labeled = np.ones((combined.shape[0] + label_h, combined.shape[1], 3),
                          dtype=np.uint8) * 255
        labeled[label_h:, :] = combined
        cv2.putText(labeled, "GROUND TRUTH FLOW", (50, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(labeled, "PREDICTED FLOW",
                    (results["pred"].shape[1] + 40, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        results["side_by_side"] = labeled

        # --- Difference heatmap ---
        diff = pred - gt
        diff_mag = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)
        dmax = np.percentile(diff_mag, 98)
        if dmax < 1e-3:
            dmax = 1.0
        diff_vis = (np.clip(diff_mag / dmax, 0, 1) * 255).astype(np.uint8)
        results["diff"] = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)

    # --- Color wheel ---
    results["wheel"] = make_color_wheel(Config.WHEEL_SIZE)

    return results


def save_visualizations(results: dict, out_dir: str, tag: str = "flow") -> dict:
    """Save all visualization arrays to disk. Returns paths dict."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, img in results.items():
        ext = ".png"
        # Wheel is RGB, rest is BGR
        if key == "wheel":
            save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            save_img = img
        path = os.path.join(out_dir, f"{tag}_{key}{ext}")
        cv2.imwrite(path, save_img)
        paths[key] = path
    return paths


# =============================================================================
# FASTAPI HELPER
# =============================================================================

def visualize_flow_api(
    pred_flow: Union[str, np.ndarray],
    gt_flow: Optional[Union[str, np.ndarray]] = None,
    out_dir: str = "/tmp/flow_viz",
    tag: str = "viz",
    step: int = None,
    arrow_scale: float = None,
) -> dict:
    """
    One-liner for your FastAPI backend.
    Accepts paths or arrays. Saves images. Returns file paths.
    """
    results = visualize_flow(pred_flow, gt_flow, step=step, arrow_scale=arrow_scale)
    return save_visualizations(results, out_dir, tag)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dense HSV Motion Vector Visualizer"
    )
    parser.add_argument("--pred", required=True, help="Predicted flow (.flo/.npy)")
    parser.add_argument("--gt", help="Ground-truth flow (.flo/.npy)")
    parser.add_argument("--out", default="./flow_viz", help="Output directory")
    parser.add_argument("--tag", default="flow", help="Filename prefix")
    parser.add_argument("--step", type=int, default=20,
                        help="Arrow spacing in pixels (lower = denser)")
    parser.add_argument("--arrow-scale", type=float, default=3.0,
                        help="Arrow length multiplier")
    parser.add_argument("--max-mag", type=float, default=None,
                        help="Shared magnitude cap (auto if omitted)")

    args = parser.parse_args()

    results = visualize_flow(
        args.pred, args.gt,
        step=args.step,
        arrow_scale=args.arrow_scale,
        max_mag=args.max_mag,
    )
    paths = save_visualizations(results, args.out, args.tag)

    print("Generated:")
    for k, v in paths.items():
        print(f"  {k:15s} -> {v}")


if __name__ == "__main__":
    main()
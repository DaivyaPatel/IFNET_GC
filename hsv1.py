#!/usr/bin/env python3
"""
Dense HSV Motion Vector Visualization Generator
================================================
Generates high-quality dense HSV optical flow visualizations with clear 
overlaid arrow vectors. Designed for frame interpolation dashboards.

Supports:
- Standard .flo optical flow files (Middlebury format)
- NumPy array files (.npy) with shape (H, W, 2)
- Side-by-side predicted vs ground-truth comparison
- High-DPI PNG output suitable for web dashboards

Usage:
    python hsv_motion_visualizer.py --pred flow_pred.flo --gt flow_gt.flo --out ./viz/
    python hsv_motion_visualizer.py --pred flow_pred.npy --gt flow_gt.npy --out ./viz/
"""

import os
import sys
import argparse
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# =============================================================================
# CONFIGURATION
# =============================================================================

class VizConfig:
    """Visualization tuning parameters."""
    # Arrow overlay settings
    ARROW_SUBSAMPLE = 24          # Draw arrow every N pixels (lower = denser)
    ARROW_SCALE = 1.0             # Multiplier for arrow length
    ARROW_WIDTH = 0.003           # Shaft width relative to plot
    ARROW_HEAD_WIDTH = 0.008      # Head width relative to plot
    ARROW_HEAD_LENGTH = 0.010     # Head length relative to plot
    ARROW_COLOR = "black"         # Arrow color (use 'white' for dark backgrounds)
    ARROW_ALPHA = 0.85            # Arrow transparency

    # HSV encoding settings
    MAX_MAG_PCTILE = 99.5         # Percentile for magnitude saturation cap
    MIN_MAG_THRESH = 0.5          # Pixels below this magnitude shown as white/gray

    # Output settings
    DPI = 200                     # Output resolution
    FIG_WIDTH = 14                # Figure width in inches
    FIG_HEIGHT_PER_ROW = 5        # Height per image row

    # Color wheel legend size
    WHEEL_SIZE = 128


# =============================================================================
# FLOW FILE I/O
# =============================================================================

def read_flo_file(path: str) -> np.ndarray:
    """
    Read a .flo file in the Middlebury optical flow format.
    Returns array of shape (H, W, 2) with dtype float32.
    """
    with open(path, "rb") as f:
        tag = struct.unpack("f", f.read(4))[0]
        if tag != 202021.25:
            raise ValueError(f"Invalid .flo file: {path} (tag={tag})")
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    flow = data.reshape((h, w, 2))
    return flow


def read_flow(path: str) -> np.ndarray:
    """
    Read optical flow from .flo or .npy file.
    Returns (H, W, 2) float32 array.
    """
    p = Path(path)
    if p.suffix == ".flo":
        return read_flo_file(path)
    elif p.suffix == ".npy":
        flow = np.load(path)
        if flow.ndim == 4 and flow.shape[0] == 1:
            flow = flow[0]
        if flow.shape[-1] != 2:
            raise ValueError(f"Flow array must have last dim=2, got shape {flow.shape}")
        return flow.astype(np.float32)
    else:
        raise ValueError(f"Unsupported flow format: {p.suffix}. Use .flo or .npy")


# =============================================================================
# DENSE HSV COLOR ENCODING
# =============================================================================

def flow_to_hsv(flow: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    """
    Convert dense optical flow to HSV color image using the standard
    optical flow color wheel (Middlebury style).

    Args:
        flow: (H, W, 2) array with (u, v) displacements.
        max_magnitude: If provided, use this as the saturation ceiling.
                       Otherwise computed from flow percentiles.

    Returns:
        (H, W, 3) uint8 BGR image (OpenCV format) ready for display/saving.
    """
    u = flow[..., 0]
    v = flow[..., 1]

    # Compute polar coordinates
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    # Determine magnitude normalization
    if max_magnitude is None:
        max_magnitude = np.percentile(mag, VizConfig.MAX_MAG_PCTILE)
        if max_magnitude < 1e-3:
            max_magnitude = 1.0

    # HSV channels
    # Hue: direction (0-180 in OpenCV HSV)
    h = ang / 2.0  # Map 0-360 -> 0-180 for OpenCV

    # Saturation: magnitude (clipped and normalized)
    s = np.clip(mag / max_magnitude, 0, 1) * 255

    # Value: constant high for visibility, dim for very low motion
    v_ch = np.ones_like(mag) * 255
    # Fade to white/gray for near-zero motion
    motion_mask = mag > VizConfig.MIN_MAG_THRESH
    v_ch = np.where(motion_mask, v_ch, 200)  # Slightly dim for static
    s = np.where(motion_mask, s, 0)          # Desaturate static -> white/gray

    hsv = np.stack([h, s, v_ch], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def make_color_wheel(size: int = 256) -> np.ndarray:
    """
    Generate the standard optical flow color wheel legend.
    Returns (size, size, 3) RGB uint8 image.
    """
    yy, xx = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    mag = np.sqrt(xx**2 + yy**2)
    ang = np.arctan2(-yy, -xx)  # Negative y for image coordinates

    # Mask outside circle
    mask = mag <= 1.0

    # HSV encoding
    h = (ang + np.pi) / (2 * np.pi) * 180  # 0-180 for OpenCV
    s = np.clip(mag, 0, 1) * 255
    v = np.ones_like(mag) * 255

    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Zero out outside circle
    rgb[~mask] = 255
    return rgb


# =============================================================================
# ARROW VECTOR OVERLAY
# =============================================================================

def overlay_arrows(ax, flow: np.ndarray, 
                   subsample: int = 16,
                   scale: float = 1.0,
                   color: str = "black",
                   alpha: float = 0.85) -> None:
    """
    Overlay clear arrow vectors on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on.
        flow: (H, W, 2) flow array.
        subsample: Draw arrow every N pixels.
        scale: Length multiplier.
        color: Arrow color.
        alpha: Arrow transparency.
    """
    h, w = flow.shape[:2]

    # Create grid
    step = max(subsample, 4)
    y_coords = np.arange(0, h, step)
    x_coords = np.arange(0, w, step)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    u = flow[yy, xx, 0] * scale
    v = flow[yy, xx, 1] * scale

    # Normalize coordinates to [0, 1] for quiver
    xx_norm = xx / w
    yy_norm = yy / h
    u_norm = u / w
    v_norm = v / h

    ax.quiver(
        xx_norm, yy_norm, u_norm, v_norm,
        color=color,
        alpha=alpha,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=VizConfig.ARROW_WIDTH,
        headwidth=3,
        headlength=4,
        headaxislength=3.5,
        minlength=0.01,
        pivot="tail",
        linewidth=0.4,
        edgecolors="white" if color == "black" else "black",
        linewidths=0.3,
    )


def overlay_arrows_on_image(img: np.ndarray, flow: np.ndarray,
                            subsample: int = 24,
                            scale: float = 2.0) -> np.ndarray:
    """
    Draw arrows directly on an image using OpenCV (alternative to matplotlib).
    Good for quick preview or when matplotlib is not needed.

    Returns:
        Image with arrows drawn (BGR uint8).
    """
    out = img.copy()
    h, w = flow.shape[:2]
    step = max(subsample, 4)

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x] * scale
            mag = np.sqrt(dx**2 + dy**2)
            if mag < VizConfig.MIN_MAG_THRESH:
                continue
            x2 = int(np.clip(x + dx, 0, w - 1))
            y2 = int(np.clip(y + dy, 0, h - 1))
            cv2.arrowedLine(
                out, (x, y), (x2, y2),
                color=(0, 0, 0),  # Black arrows
                thickness=2,
                tipLength=0.3,
                line_type=cv2.LINE_AA,
            )
            # White outline for contrast
            cv2.arrowedLine(
                out, (x, y), (x2, y2),
                color=(255, 255, 255),
                thickness=1,
                tipLength=0.3,
                line_type=cv2.LINE_AA,
            )
    return out


# =============================================================================
# MAIN VISUALIZATION COMPOSER
# =============================================================================

def generate_flow_visualization(
    pred_flow_path: Optional[str] = None,
    gt_flow_path: Optional[str] = None,
    frame_path: Optional[str] = None,  # Optional background frame
    output_dir: str = "./flow_viz",
    tag: str = "frame",
    max_magnitude: Optional[float] = None,
) -> dict:
    """
    Generate high-quality HSV motion vector visualization.

    Args:
        pred_flow_path: Path to predicted/interpolated flow.
        gt_flow_path: Path to ground-truth flow.
        frame_path: Optional background image (PNG/JPG) for context.
        output_dir: Where to save outputs.
        tag: Filename prefix.
        max_magnitude: Shared magnitude cap for consistent colors across frames.

    Returns:
        Dictionary with paths to generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Load flows
    flows = {}
    if pred_flow_path:
        flows["Interpolated"] = read_flow(pred_flow_path)
    if gt_flow_path:
        flows["Ground Truth"] = read_flow(gt_flow_path)

    if not flows:
        raise ValueError("At least one flow file (pred or gt) must be provided.")

    # Determine shared max magnitude for consistent coloring
    if max_magnitude is None:
        all_mags = []
        for fl in flows.values():
            u, v = fl[..., 0], fl[..., 1]
            mag, _ = cv2.cartToPolar(u, v)
            all_mags.append(mag)
        max_magnitude = np.percentile(np.concatenate([m.ravel() for m in all_mags]), 
                                      VizConfig.MAX_MAG_PCTILE)
        if max_magnitude < 1e-3:
            max_magnitude = 1.0

    # Load optional background frame
    bg_frame = None
    if frame_path and os.path.exists(frame_path):
        bg_frame = cv2.imread(frame_path)
        bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # 1) Individual dense HSV + arrows (Matplotlib high-quality)
    # -------------------------------------------------------------------------
    n_plots = len(flows)
    fig, axes = plt.subplots(1, n_plots, figsize=(VizConfig.FIG_WIDTH, 
                                                   VizConfig.FIG_HEIGHT_PER_ROW))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, flow) in zip(axes, flows.items()):
        hsv_bgr = flow_to_hsv(flow, max_magnitude=max_magnitude)
        hsv_rgb = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)

        ax.imshow(hsv_rgb)
        ax.set_title(f"{name}\nDense HSV Motion Field", fontsize=12, fontweight="bold")
        ax.axis("off")

        # Overlay arrows
        overlay_arrows(
            ax, flow,
            subsample=VizConfig.ARROW_SUBSAMPLE,
            scale=VizConfig.ARROW_SCALE,
            color=VizConfig.ARROW_COLOR,
            alpha=VizConfig.ARROW_ALPHA,
        )

    plt.tight_layout()
    dense_path = os.path.join(output_dir, f"{tag}_dense_hsv_arrows.png")
    fig.savefig(dense_path, dpi=VizConfig.DPI, bbox_inches="tight", 
                pad_inches=0.1, facecolor="white")
    plt.close(fig)
    paths["dense_hsv_arrows"] = dense_path

    # -------------------------------------------------------------------------
    # 2) Dense HSV with arrows overlaid on background frame (if provided)
    # -------------------------------------------------------------------------
    if bg_frame is not None:
        fig, axes = plt.subplots(1, n_plots, figsize=(VizConfig.FIG_WIDTH, 
                                                       VizConfig.FIG_HEIGHT_PER_ROW))
        if n_plots == 1:
            axes = [axes]

        for ax, (name, flow) in zip(axes, flows.items()):
            # Resize frame to flow if needed
            if bg_frame.shape[:2] != flow.shape[:2]:
                bg_resized = cv2.resize(bg_frame, (flow.shape[1], flow.shape[0]))
            else:
                bg_resized = bg_frame

            # Blend HSV with background
            hsv_bgr = flow_to_hsv(flow, max_magnitude=max_magnitude)
            hsv_rgb = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)
            blended = cv2.addWeighted(bg_resized, 0.4, hsv_rgb, 0.6, 0)

            ax.imshow(blended)
            ax.set_title(f"{name}\nOverlay on Frame", fontsize=12, fontweight="bold")
            ax.axis("off")
            overlay_arrows(ax, flow, subsample=VizConfig.ARROW_SUBSAMPLE,
                           scale=VizConfig.ARROW_SCALE, color="yellow",
                           alpha=0.9)

        plt.tight_layout()
        overlay_path = os.path.join(output_dir, f"{tag}_overlay_arrows.png")
        fig.savefig(overlay_path, dpi=VizConfig.DPI, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths["overlay_arrows"] = overlay_path

    # -------------------------------------------------------------------------
    # 3) Side-by-side comparison with difference map (if both pred & gt)
    # -------------------------------------------------------------------------
    if "Interpolated" in flows and "Ground Truth" in flows:
        pred = flows["Interpolated"]
        gt = flows["Ground Truth"]

        # Ensure same shape
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

        diff = pred - gt
        diff_mag = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)

        fig, axes = plt.subplots(2, 3, figsize=(VizConfig.FIG_WIDTH, 
                                                 VizConfig.FIG_HEIGHT_PER_ROW * 1.8))

        # Row 1: Individual HSV fields
        for idx, (name, flow) in enumerate([("Interpolated", pred), ("Ground Truth", gt)]):
            hsv_bgr = flow_to_hsv(flow, max_magnitude=max_magnitude)
            hsv_rgb = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2RGB)
            axes[0, idx].imshow(hsv_rgb)
            axes[0, idx].set_title(f"{name} HSV", fontsize=11, fontweight="bold")
            axes[0, idx].axis("off")
            overlay_arrows(axes[0, idx], flow, subsample=VizConfig.ARROW_SUBSAMPLE,
                           scale=VizConfig.ARROW_SCALE)

        # Difference HSV (encode error direction)
        diff_hsv = flow_to_hsv(diff, max_magnitude=np.percentile(diff_mag, 98))
        axes[0, 2].imshow(cv2.cvtColor(diff_hsv, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Difference (Pred - GT)", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

        # Row 2: Magnitude heatmaps
        for idx, (name, flow) in enumerate([("Interpolated", pred), ("Ground Truth", gt)]):
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            im = axes[1, idx].imshow(mag, cmap="turbo", interpolation="nearest")
            axes[1, idx].set_title(f"{name} Magnitude", fontsize=11, fontweight="bold")
            axes[1, idx].axis("off")
            plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

        im = axes[1, 2].imshow(diff_mag, cmap="hot", interpolation="nearest")
        axes[1, 2].set_title("Magnitude Error", fontsize=11, fontweight="bold")
        axes[1, 2].axis("off")
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        compare_path = os.path.join(output_dir, f"{tag}_comparison.png")
        fig.savefig(compare_path, dpi=VizConfig.DPI, bbox_inches="tight",
                    pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths["comparison"] = compare_path

    # -------------------------------------------------------------------------
    # 4) Color wheel legend (standalone)
    # -------------------------------------------------------------------------
    wheel = make_color_wheel(size=VizConfig.WHEEL_SIZE)
    wheel_path = os.path.join(output_dir, f"{tag}_colorwheel.png")
    plt.imsave(wheel_path, wheel)
    paths["colorwheel"] = wheel_path

    return paths


# =============================================================================
# FASTAPI / BACKEND INTEGRATION HELPER
# =============================================================================

def visualize_flow_for_api(
    pred_flow: Union[str, np.ndarray],
    gt_flow: Optional[Union[str, np.ndarray]] = None,
    reference_frame: Optional[Union[str, np.ndarray]] = None,
    output_dir: str = "/tmp/flow_viz",
    tag: str = "viz",
) -> dict:
    """
    Backend-friendly wrapper for API integration.
    Accepts file paths OR numpy arrays.

    Returns dict with file paths for the dashboard to serve.
    """
    import tempfile

    # Save arrays to temp files if needed
    temp_files = []
    def _ensure_path(data, suffix):
        if isinstance(data, str):
            return data
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        np.save(tmp.name, data)
        temp_files.append(tmp.name)
        return tmp.name

    pred_path = _ensure_path(pred_flow, ".npy") if pred_flow is not None else None
    gt_path = _ensure_path(gt_flow, ".npy") if gt_flow is not None else None
    frame_path = reference_frame if isinstance(reference_frame, str) else None
    if isinstance(reference_frame, np.ndarray):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, reference_frame)
        temp_files.append(tmp.name)
        frame_path = tmp.name

    try:
        result = generate_flow_visualization(
            pred_flow_path=pred_path,
            gt_flow_path=gt_path,
            frame_path=frame_path,
            output_dir=output_dir,
            tag=tag,
        )
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dense HSV motion vector visualizations"
    )
    parser.add_argument("--pred", type=str, help="Path to predicted flow (.flo or .npy)")
    parser.add_argument("--gt", type=str, help="Path to ground-truth flow (.flo or .npy)")
    parser.add_argument("--frame", type=str, help="Optional reference frame image")
    parser.add_argument("--out", type=str, default="./flow_viz", help="Output directory")
    parser.add_argument("--tag", type=str, default="flow", help="Output filename prefix")
    parser.add_argument("--subsample", type=int, default=24, 
                        help="Arrow subsample step (lower = denser arrows)")
    parser.add_argument("--arrow-scale", type=float, default=1.0,
                        help="Scale factor for arrow length")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")

    args = parser.parse_args()

    VizConfig.ARROW_SUBSAMPLE = args.subsample
    VizConfig.ARROW_SCALE = args.arrow_scale
    VizConfig.DPI = args.dpi

    result = generate_flow_visualization(
        pred_flow_path=args.pred,
        gt_flow_path=args.gt,
        frame_path=args.frame,
        output_dir=args.out,
        tag=args.tag,
    )

    print("Generated files:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
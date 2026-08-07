"""
app/core/optical_flow.py

Motion analysis module — independent post-processing on top of the
interpolation pipeline. No dependency on RIFE or any interpolation logic.

Public API:
    generate_optical_flow(image1_path, image2_path, estimator=None) -> bytes

Operates on already-composited RGB PNG/JPEG images (the same T0 / T_real /
T_interpolated images used elsewhere in the pipeline) — NOT on raw
satellite .nc / .h5 files. Those are preprocessed upstream (see the RGB
composite builder in this codebase) before ever reaching this module.

Design:
    - OpticalFlowEstimator is an abstract strategy interface. Today's
      concrete implementation is FarnebackFlowEstimator (OpenCV). Future
      algorithms (DIS, RAFT, PWC-Net, DeepFlow) can be added as new
      subclasses without changing generate_optical_flow's signature or
      any other part of the backend.
    - flow_to_hsv() and encode_png() are algorithm-agnostic — they operate
      on the raw (H, W, 2) flow field regardless of which estimator
      produced it.
"""

from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class OpticalFlowError(Exception):
    """Base exception for all optical flow module errors."""


class ImageLoadError(OpticalFlowError):
    """Raised when an input image cannot be located, fetched, or decoded."""


class InvalidImageError(OpticalFlowError):
    """Raised when an image is loaded but is empty, corrupt, or unusable."""


class FlowComputationError(OpticalFlowError):
    """Raised when dense optical flow computation itself fails."""


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

_URL_PREFIXES: Final[tuple[str, ...]] = ("http://", "https://")


def _is_url(path: str) -> bool:
    """Check whether a given path string is a URL rather than a local path."""
    return path.startswith(_URL_PREFIXES)


def _read_bytes_from_url(url: str, timeout: int = 10) -> bytes:
    """
    Fetch raw bytes from a remote URL.

    Raises:
        ImageLoadError: if the URL cannot be reached or read.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise ImageLoadError(f"Failed to fetch image from URL '{url}': {exc}") from exc


def _read_bytes_from_local(path: str) -> bytes:
    """
    Read raw bytes from a local filesystem path.

    Raises:
        ImageLoadError: if the file does not exist or cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageLoadError(f"Image path does not exist or is not a file: '{path}'")
    try:
        return file_path.read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"Failed to read image file '{path}': {exc}") from exc


def load_image(path: str) -> np.ndarray:
    """
    Load an image (local path or URL) as a BGR NumPy array.

    Supports PNG and JPEG. Works with both local filesystem paths and
    remote URLs (http/https).

    Args:
        path: Local file path or URL pointing to the image.

    Returns:
        np.ndarray: Decoded image in BGR format, shape (H, W, 3).

    Raises:
        ImageLoadError: If the image cannot be located or fetched.
        InvalidImageError: If the bytes exist but cannot be decoded, or
            decode to an empty image.
    """
    if not path or not isinstance(path, str):
        raise ImageLoadError(f"Invalid image path provided: {path!r}")

    raw_bytes = (
        _read_bytes_from_url(path) if _is_url(path) else _read_bytes_from_local(path)
    )

    if not raw_bytes:
        raise InvalidImageError(f"Image source '{path}' returned empty content.")

    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise InvalidImageError(
            f"Could not decode image from '{path}'. "
            f"Ensure it is a valid PNG or JPEG file."
        )

    return image


def ensure_same_size(image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure two images share the same spatial dimensions, resizing the
    second image to match the first if needed (required by Farneback
    and most dense flow algorithms).

    Args:
        image1: Reference image, BGR, shape (H, W, 3).
        image2: Image to align, BGR, shape (H2, W2, 3).

    Returns:
        Tuple of (image1, image2_resized), both with matching shape.
    """
    if image1.shape[:2] != image2.shape[:2]:
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_LINEAR)
    return image1, image2


# --------------------------------------------------------------------------- #
# Optical Flow Estimators
# --------------------------------------------------------------------------- #

class OpticalFlowEstimator(ABC):
    """
    Abstract interface for dense optical flow algorithms.

    Any concrete estimator (Farneback, DIS, RAFT, PWC-Net, DeepFlow, ...)
    must implement `compute_flow` and return a raw flow field. This keeps
    `generate_optical_flow()` and `flow_to_hsv()` completely agnostic to
    which algorithm produced the flow — swapping algorithms means writing
    a new subclass, nothing else in the backend changes.
    """

    @abstractmethod
    def compute_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two images.

        Args:
            image1: First frame, BGR, shape (H, W, 3).
            image2: Second frame, BGR, shape (H, W, 3). Must match image1's
                spatial dimensions (caller is responsible for resizing,
                see `ensure_same_size`).

        Returns:
            np.ndarray: Dense flow field of shape (H, W, 2), where the last
                axis holds (dx, dy) per-pixel displacement.

        Raises:
            FlowComputationError: If flow computation fails.
        """
        raise NotImplementedError


class FarnebackFlowEstimator(OpticalFlowEstimator):
    """
    Dense optical flow using Gunnar Farneback's polynomial expansion
    algorithm (cv2.calcOpticalFlowFarneback).

    Well suited as a fast, dependency-light baseline for smooth, gradual
    motion such as cloud displacement across satellite frames. Parameters
    are exposed via the constructor so they can be tuned without touching
    the algorithm logic.
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        flags: int = 0,
    ) -> None:
        """
        Args:
            pyr_scale: Image scale (<1) to build pyramids for each image.
            levels: Number of pyramid layers, including the initial image.
            winsize: Averaging window size. Larger values catch broader,
                smoother motion (good for gradual cloud displacement) at
                the cost of finer detail.
            iterations: Number of iterations at each pyramid level.
            poly_n: Size of the pixel neighborhood for polynomial expansion.
            poly_sigma: Standard deviation of the Gaussian used to smooth
                derivatives for the polynomial expansion.
            flags: Operation flags for cv2.calcOpticalFlowFarneback.
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def compute_flow(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two BGR images using Farneback.

        Raises:
            FlowComputationError: If either image is invalid or OpenCV
                fails to compute the flow field.
        """
        if image1 is None or image2 is None or image1.size == 0 or image2.size == 0:
            raise FlowComputationError("Cannot compute flow: one or both images are empty.")

        if image1.shape[:2] != image2.shape[:2]:
            raise FlowComputationError(
                f"Image shape mismatch for flow computation: "
                f"{image1.shape[:2]} vs {image2.shape[:2]}. "
                f"Call ensure_same_size() before compute_flow()."
            )

        try:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                gray1,
                gray2,
                None,
                self.pyr_scale,
                self.levels,
                self.winsize,
                self.iterations,
                self.poly_n,
                self.poly_sigma,
                self.flags,
            )
        except cv2.error as exc:
            raise FlowComputationError(f"Farneback optical flow computation failed: {exc}") from exc

        if flow is None:
            raise FlowComputationError("Farneback optical flow returned no result.")

        return flow


# --------------------------------------------------------------------------- #
# HSV Visualization
# --------------------------------------------------------------------------- #

def flow_to_hsv(flow: np.ndarray, max_magnitude: float = 20.0) -> np.ndarray:
    """
    Convert a raw dense optical flow field into an HSV-based BGR
    visualization image.

    Mapping:
        Hue        <- motion direction (angle of the flow vector)
        Saturation <- always 255 (full color intensity)
        Value      <- motion magnitude, normalized against a FIXED scale
                      (not per-image min-max), so that two flow images
                      produced from different frame pairs remain visually
                      comparable in brightness when viewed side by side
                      (e.g. Ground Truth Flow vs Predicted Flow on the
                      dashboard).

    Args:
        flow: Dense flow field, shape (H, W, 2), from any
            OpticalFlowEstimator.compute_flow() implementation.
        max_magnitude: Displacement (in pixels) that maps to full
            brightness (255). Displacement above this value is clipped.
            Tune this based on the typical pixel displacement observed
            in your satellite frame pairs. Default 20.0 is a reasonable
            starting point for gradual cloud motion.

    Returns:
        np.ndarray: BGR image, shape (H, W, 3), dtype uint8 — ready for
            PNG encoding.

    Raises:
        FlowComputationError: If the flow field is empty, malformed, or
            max_magnitude is not positive.
    """
    if flow is None or flow.size == 0:
        raise FlowComputationError("Cannot visualize flow: flow field is empty.")

    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise FlowComputationError(
            f"Expected flow field of shape (H, W, 2), got shape {flow.shape}."
        )

    if max_magnitude <= 0:
        raise FlowComputationError(f"max_magnitude must be positive, got {max_magnitude}.")

    h, w = flow.shape[:2]
    
    # Scale down by 50% to reduce huge image size
    new_w, new_h = w // 2, h // 2
    flow_small = cv2.resize(flow, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    magnitude, angle = cv2.cartToPolar(flow_small[..., 0], flow_small[..., 1], angleInDegrees=True)
    
    # Black background
    bgr = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # Draw vector field (arrows) colored by HSV
    step = 16
    vector_scale = 1.0  # Scale vectors
    
    for y in range(step // 2, new_h, step):
        for x in range(step // 2, new_w, step):
            mag = magnitude[y, x]
            ang = angle[y, x]
            dx, dy = flow_small[y, x]
            
            # only draw if magnitude is large enough to be visible
            if mag > 1.0:
                # Calculate color
                hue = int(ang / 2)
                val = int(min((mag / max_magnitude) * 255.0, 255.0))
                
                # Convert this specific pixel's HSV to BGR
                color_hsv = np.uint8([[[hue, 255, val]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                
                dx, dy = dx * vector_scale, dy * vector_scale
                pt1 = (x, y)
                pt2 = (int(x + dx), int(y + dy))
                cv2.arrowedLine(bgr, pt1, pt2, color, 1, cv2.LINE_AA, tipLength=0.25)
                
    return bgr


def encode_png(image: np.ndarray) -> bytes:
    """
    Encode a BGR image array as PNG bytes, without writing to disk.

    Args:
        image: BGR image, shape (H, W, 3), dtype uint8.

    Returns:
        bytes: PNG-encoded image data.

    Raises:
        FlowComputationError: If encoding fails.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise FlowComputationError("Failed to encode image as PNG.")
    return buffer.tobytes()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_optical_flow(
    image1_path: str,
    image2_path: str,
    estimator: Optional[OpticalFlowEstimator] = None,
    max_magnitude: float = 20.0,
) -> bytes:
    """
    Generate an HSV dense-optical-flow visualization between two images.

    This is the single public entry point for the motion analysis module.
    It is independent of the interpolation pipeline: it does not import
    or call RIFE, and does not modify any interpolation logic. It only
    consumes already-composited RGB PNG/JPEG images (local paths or URLs)
    — the same images used elsewhere in the pipeline (T0, T_real,
    T_interpolated, Tn).

    Pipeline:
        load image1, load image2
        -> ensure equal spatial size
        -> estimator.compute_flow()   (Farneback by default)
        -> flow_to_hsv()
        -> encode_png()
        -> return PNG bytes (no temp files written)

    Args:
        image1_path: Local path or URL to the first (reference) image.
        image2_path: Local path or URL to the second (target) image.
        estimator: Optical flow algorithm to use. Defaults to
            FarnebackFlowEstimator() if not provided. Swap in a different
            OpticalFlowEstimator subclass (e.g. a future RAFTFlowEstimator)
            to change the algorithm without changing this function's
            signature or call sites.
        max_magnitude: Passed through to flow_to_hsv() — the pixel
            displacement that maps to full brightness in the
            visualization. Keep this consistent across calls that will
            be displayed side by side (e.g. Ground Truth Flow vs
            Predicted Flow) so brightness remains visually comparable.

    Returns:
        bytes: PNG-encoded HSV optical flow visualization.

    Raises:
        ImageLoadError: If either image cannot be located or fetched.
        InvalidImageError: If either image is empty or cannot be decoded.
        FlowComputationError: If flow computation or encoding fails.

    Example:
        >>> flow_png_bytes = generate_optical_flow("samples/T0.png", "samples/T_real.png")
        >>> with open("output/HSV_FLOW_REAL.png", "wb") as f:
        ...     f.write(flow_png_bytes)
    """
    active_estimator = estimator if estimator is not None else FarnebackFlowEstimator()

    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    image1, image2 = ensure_same_size(image1, image2)

    flow = active_estimator.compute_flow(image1, image2)

    # Use the new hsv_visualizer (hsv2) for high-contrast arrows
    from app.core.hsv_visualizer import visualize_flow_api
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = visualize_flow_api(
            pred_flow=flow,
            out_dir=tmpdir,
            tag="flow"
        )
        with open(paths["pred"], "rb") as f:
            return f.read()

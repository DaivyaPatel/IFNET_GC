import io
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray
import lpips
import torch

# Initialize LPIPS model once (using AlexNet backbone which is standard)
lpips_loss_fn = None



def _load_as_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = sobel(gray, axis=0)
    gy = sobel(gray, axis=1)
    return np.hypot(gx, gy)


def _fsim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Simplified FSIM (Feature Similarity Index), using gradient magnitude
    as the feature map. This is an approximation of the full published
    FSIM (which also uses phase congruency) — accurate enough for
    relative comparisons in this project, but not a byte-for-byte match
    to reference FSIM implementations.
    """
    gray_a = rgb2gray(img_a) * 255.0
    gray_b = rgb2gray(img_b) * 255.0

    gm_a = _gradient_magnitude(gray_a)
    gm_b = _gradient_magnitude(gray_b)

    T = 160.0  # standard FSIM constant for gradient similarity
    gm_similarity = (2 * gm_a * gm_b + T) / (gm_a ** 2 + gm_b ** 2 + T)

    weight = np.maximum(gm_a, gm_b)
    if weight.sum() == 0:
        return 1.0

    return float((gm_similarity * weight).sum() / weight.sum())


def compute_metrics(image_bytes_a: bytes, image_bytes_b: bytes) -> dict:
    """
    Compares two images and returns PSNR, SSIM, RMSE, MAE, FSIM, and LPIPS.
    Both images are resized to match the smaller of the two if their
    dimensions differ, so the comparison can still run.
    """
    global lpips_loss_fn
    if lpips_loss_fn is None:
        lpips_loss_fn = lpips.LPIPS(net='alex')
    img_a = _load_as_array(image_bytes_a)
    img_b = _load_as_array(image_bytes_b)

    if img_a.shape != img_b.shape:
        target_h = min(img_a.shape[0], img_b.shape[0])
        target_w = min(img_a.shape[1], img_b.shape[1])
        img_a = np.array(Image.fromarray(img_a).resize((target_w, target_h)))
        img_b = np.array(Image.fromarray(img_b).resize((target_w, target_h)))

    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))))
    psnr = float(peak_signal_noise_ratio(img_a, img_b, data_range=255))
    ssim = float(structural_similarity(img_a, img_b, channel_axis=2, data_range=255))
    fsim = _fsim(img_a, img_b)

    # Compute LPIPS
    # Convert numpy [0, 255] (H, W, C) to PyTorch [-1, 1] (1, C, H, W)
    img_a_tensor = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
    img_b_tensor = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = float(lpips_loss_fn(img_a_tensor, img_b_tensor).item())

    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse": rmse,
        "mae": mae,
        "fsim": fsim,
        "lpips": lpips_val,
    }
import numpy as np
import cv2
from PIL import Image
import io

print("Testing difference map creation...")
try:
    img1 = np.zeros((256, 256, 3), dtype=np.uint8)
    img2 = np.full((256, 256, 3), 100, dtype=np.uint8)

    diff = cv2.absdiff(img1, img2)
    gray_diff = np.max(diff, axis=2).astype(np.uint8)
    
    r = np.full_like(gray_diff, 255)
    g = 255 - gray_diff
    b = 255 - gray_diff
    
    heatmap_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    buffer = io.BytesIO()
    Image.fromarray(heatmap_rgb).save(buffer, format="PNG")
    print("Success!")
except Exception as e:
    print("Caught Exception:", e)

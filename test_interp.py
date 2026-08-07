import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'BAH-backend'))
from app.ml.run_interpolation import interpolate
import numpy as np

# Create dummy images and BT fields
bt13_0 = np.full((256, 256), 250.0, dtype=np.float32)
bt13_1 = np.full((256, 256), 250.0, dtype=np.float32)
bt8_0 = np.full((256, 256), 220.0, dtype=np.float32)
bt8_1 = np.full((256, 256), 220.0, dtype=np.float32)

from app.ml.run_interpolation import run_pysteps, make_rgb

print("Testing Linear...")
try:
    tir_interp = (bt13_0 + bt13_1) / 2.0
    wv_interp = (bt8_0 + bt8_1) / 2.0
    pred_linear = make_rgb(tir_interp, wv_interp)
    print("Linear OK, shape:", pred_linear.shape)
except Exception as e:
    print("Linear Failed:", e)

print("Testing PySTEPS...")
try:
    pred_pysteps = run_pysteps(bt13_0, bt8_0, bt13_1, bt8_1)[0]
    print("PySTEPS OK, shape:", pred_pysteps.shape)
except Exception as e:
    print("PySTEPS Failed:", e)

print("Testing RIFE...")
try:
    from app.ml.train_log.RIFE_HDv3 import Model as RifeModel
    import torch
    rife_model = RifeModel(local_rank=-1)
    rife_model.load_model(os.path.join(os.getcwd(), "BAH-backend", "app", "ml", "train_log"), rank=-1)
    rife_model.eval()
    img0 = torch.zeros((1, 3, 256, 256))
    img1 = torch.zeros((1, 3, 256, 256))
    final = rife_model.inference(img0, img1, scale=1.0)
    print("RIFE OK, shape:", final[0].shape)
except Exception as e:
    print("RIFE Failed:", e)

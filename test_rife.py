import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'BAH-backend'))

from app.ml.train_log.RIFE_HDv3 import Model as RifeModel
import torch

def test_linear():
    import numpy as np
    from app.ml.evolution_model import make_rgb
    
    bt13_0 = np.random.rand(256, 256).astype(np.float32) * 100 + 200
    bt8_0 = np.random.rand(256, 256).astype(np.float32) * 100 + 200
    bt13_1 = np.random.rand(256, 256).astype(np.float32) * 100 + 200
    bt8_1 = np.random.rand(256, 256).astype(np.float32) * 100 + 200
    
    tir_interp = (bt13_0 + bt13_1) / 2.0
    wv_interp = (bt8_0 + bt8_1) / 2.0
    pred = make_rgb(tir_interp, wv_interp)
    print("Linear output shape:", pred.shape)
    
def test_rife():
    import pathlib
    ROOT_DIR = pathlib.Path(__file__).resolve().parent
    rife_model = RifeModel(local_rank=-1)
    
    img0 = torch.rand(1, 3, 256, 256).cuda() if torch.cuda.is_available() else torch.rand(1, 3, 256, 256)
    img1 = torch.rand(1, 3, 256, 256).cuda() if torch.cuda.is_available() else torch.rand(1, 3, 256, 256)
    
    rife_path = os.path.join(os.getcwd(), "BAH-backend", "app", "ml", "train_log")
    rife_model.load_model(rife_path, rank=-1)
    rife_model.eval()
    
    try:
        final = rife_model.inference(img0, img1, scale=1.0)
        print("Rife output shape:", final.shape)
    except Exception as e:
        print("Rife inference failed:", e)

if __name__ == "__main__":
    test_linear()
    test_rife()

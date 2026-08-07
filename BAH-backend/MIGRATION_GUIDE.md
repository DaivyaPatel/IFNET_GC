Fine-Tuned Backend Integration — EvolutionRefinementNet Architecture

Architecture Overview

The actual model is NOT IFNet_GC with gap_embed inside IFBlock.
It is a two-stage architecture:
base_flownet = Frozen stock RIFE IFNet (original bidirectional flow, no gap conditioning)
refine = EvolutionRefinementNet (separate residual branch, zero-initialized head)
ema_refine = EMA copy of refine (used by default for inference)
Inference Contract

Python

final, base_merged, residual = model.inference(
    img0, img1,           # composite RGB tensors [1,3,H,W]
    tir0, tir1,           # normalized TIR bands [1,1,H,W]  (R channel of composite)
    wv0, wv1,             # normalized WV bands  [1,1,H,W]  (G channel of composite)
    gap=gap_hours,        # half_gap_minutes / 60.0
    scale=1.0,
    use_ema=True
)
The refine branch inputs are:
base_merged: frozen base output (3 ch)
flow: final flow field from base (4 ch)
tir0, tir1: raw normalized BT13 for img0/img1 (1 ch each)
wv0, wv1: raw normalized BT8 for img0/img1 (1 ch each)
gap: temporal gap embedding (broadcast to 16 ch spatial map)
Total input to refine stem: 3 + 4 + 1 + 1 + 1 + 1 + 16 = 27 channels
Checkpoint Format (best_checkpoint.pth)

Python

{
    "epoch": int,
    "best_psnr": float,
    "prev_epoch_loss": float,
    "base_flownet": state_dict,   # loads into IFNet strict=True
    "refine": state_dict,         # loads into EvolutionRefinementNet
    "ema_refine": state_dict,     # EMA copy (preferred for inference)
    "optimizer": ...,
    "scaler": ...
}
Files Changed

Table


File	Action	Description
app/ml/model/IFNET_GC.py	Create	New model: frozen IFNet + EvolutionRefinementNet + GapEmbedding
app/ml/run_interpolation.py	Replace	Loads .pth checkpoint, extracts R/G channels from composites as tir/wv, calls new inference
app/core/preprocessing.py	Unchanged	RGB composite + metadata extraction (still valid)
app/routes/experiment.py	Unchanged	/upload extracts gap, /run passes it (still valid)
app/database/models.py	Unchanged	gap_minutes column already added
Deployment Steps

Step 1: Place the checkpoint

bash

mkdir -p app/ml/checkpoints
cp "best_checkpoint (2).pth" app/ml/checkpoints/best_checkpoint.pth
Or set env var:
bash

export RIFE_GC_CHECKPOINT_PATH="/path/to/your/checkpoint.pth"
Step 2: Remove old model files

bash

# Delete the old GC model (wrong architecture)
rm -f app/ml/model/RIFE_HDv3_GC.py

# The old stock RIFE files are also no longer needed
rm -f app/ml/model/RIFE_HDv3.py
rm -f app/ml/model/IFNet_HDv3.py
rm -f app/ml/model/warplayer.py
Step 3: Copy new files

bash

cp IFNET_GC.py app/ml/model/
cp run_interpolation.py app/ml/
Step 4: Database

If you haven't already added gap_minutes:
sql

ALTER TABLE public.experiments ADD COLUMN gap_minutes double precision;
How TIR/WV Bands Are Provided

The backend does not store separate raw band files. Instead:
/upload builds composite PNGs (R=TIR, G=WV, B=avg) and uploads them to Supabase
/run downloads the composite PNGs and extracts:
tir = composite[:, 0:1, :, :] (R channel)
wv  = composite[:, 1:2, :, :] (G channel)
This is mathematically identical to the training script's:
Python

tir = normalize_bt(bt13, BT13_MIN, BT13_MAX)
wv  = normalize_bt(bt8,  BT8_MIN,  BT8_MAX)
The only difference is 8-bit PNG quantization (±1/255 ≈ ±0.004), which is negligible for inference.
API Contract

/upload — unchanged

Still accepts 4 named file fields: t0_tir, t0_wv, t1_tir, t1_wv
Response:
JSON

{
  "message": "Satellite files processed and composites uploaded successfully",
  "gap_minutes": 15.0
}
/run — unchanged

No body parameters.
Response:
JSON

{
  "message": "Interpolation completed",
  "generated_image_url": "...",
  "gap_minutes": 15.0
}
Testing the Integration

bash

# 1. Create experiment
POST /experiments/?model_name=EvolutionRefinement&model_version=ft_v2

# 2. Upload 4 raw satellite files
POST /experiments/{id}/upload
  FormData: t0_tir, t0_wv, t1_tir, t1_wv

# 3. Run interpolation
POST /experiments/{id}/run

# 4. Compare with ground truth
POST /experiments/{id}/compare
  FormData: real_image
File Map

plain

app/
├── ml/
│   ├── model/
│   │   └── IFNET_GC.py          ← NEW (frozen IFNet + EvolutionRefinementNet)
│   ├── run_interpolation.py      ← REPLACED (extracts R/G channels, calls new inference)
│   └── checkpoints/
│       └── best_checkpoint.pth   ← YOUR .pth FILE HERE
├── core/
│   └── preprocessing.py          ← UNCHANGED (RGB composite + metadata extraction)
├── routes/
│   └── experiment.py             ← UNCHANGED (gap extraction + passing)
└── database/
    └── models.py                 ← UNCHANGED (gap_minutes column)
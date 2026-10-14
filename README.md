# IFNET_GC: Satellite Frame Interpolation 

---

## 📌 Project Summary

Geostationary meteorological satellites such as **ISRO's INSAT-3D/3DR/3DS** and **NOAA's GOES-16/19** observe Earth at discrete scan intervals (typically 15 to 30 minutes, or longer during scan anomalies). However, critical high-impact atmospheric phenomena—such as **cyclone eye formation, rapid cloud condensation/evaporation, sudden flood surges, and fog dissipation**—evolve dynamically on sub-15-minute scales.

Standard computer-vision frame interpolation models (like stock RIFE or optical-flow advection) assume **linear motion rigidity and photometric constancy**. These assumptions catastrophically fail in meteorology because clouds undergo continuous non-rigid fluid deformation, convection, phase changes, and non-linear diurnal heating.

**IFNET_GC** solves this with a specialized **two-stage hybrid neural architecture**:
1. **Frozen Base Flownet (RIFE IFNet HDv3)**: Extracts multi-scale coarse-to-fine bidirectional flow fields and initial occlusion blend masks.
2. **EvolutionRefinementNet**: A 12-channel deep residual network that receives the base prediction, flow fields, confidence mask, and **raw physical thermal infrared (TIR) and water vapor (WV)** brightness temperature bands. It computes a localized residual correction matrix scaled by a learned per-channel gain parameter.
3. **Gap Map Spatial Conditioning (`gap_map`)**: Broadcasts arbitrary scalar time fractions into a dense spatial tensor, enabling variable time-gap interpolation and **hierarchical recursive halving** (generating smooth sequences at 15 min $\rightarrow$ 7.5 min $\rightarrow$ 3.75 min intervals).

The system includes a production-ready **FastAPI backend**, an enterprise **React 19 + Three.js interactive dashboard**, geospatial projection tools, and dedicated analytical pipelines for **tropical cyclones, floods, and fog**.

---

## 🌟 Key Capabilities & Scientific Innovations

### 1. Multi-Spectral Evolution Refinement
Unlike generic RGB interpolators, IFNET_GC directly injects normalized physical radiometric measurements into its refinement stem:
* **Clean Infrared Window (TIR1 / Band 13, $\sim 10.3\text{--}10.8\,\mu\text{m}$)**: Captures cloud-top temperatures, convective towers, and surface thermal radiation.
* **Upper-Level Water Vapor (WV / Band 8, $\sim 6.2\text{--}6.7\,\mu\text{m}$)**: Reveals atmospheric moisture advection, jet streams, and vorticity dynamics even in cloud-free zones.

### 2. Tropical Cyclone Tracking & Kinematics
* **Automated Eye Center Detection**: Employs a multi-tier fallback algorithm:
  1. *Warmest Local Maximum* inside the Central Dense Overcast (CDO) cold envelope.
  2. *Minimum-Variance Local Patch* for calm eye structures.
  3. *Motion Extrapolation & Centroid Fallback* for disorganized depressions.
* **Geodesic Trajectory & Speed Estimation**: Converts pixel coordinates to true geographical coordinates via `pyproj` / geostationary satellite projection and computes great-circle distance (Haversine formula), ground speed ($\text{km/h}$), and 16-point compass heading (e.g., WNW, NNE).

### 3. Dedicated Flood & Inundation Monitoring (`Floodscript.py`)
Computes four calibrated multi-spectral composites to isolate standing water and surface changes:
* **Day Water / Natural Flood**: $\text{VIS} - \text{VIS} - \text{SWIR}_{\text{inv}}$ (accentuates sediment-rich floodwater).
* **Pseudo-NDWI**: Normalized Difference Water Index using visible and shortwave-infrared bands.
* **Day Microphysics**: $\text{VIS} - \text{SWIR} - \text{TIR1}_{\text{inv}}$ (discriminates water vs. ice cloud phases).
* **Night Rain / Flood Proxy**: Brightness Temperature Difference ($\text{MIR} - \text{TIR1}$) combined with split-window channels.

### 4. Solar Zenith Angle-Guided Fog Analysis (`fogtestingscript.py`)
* **Solar Geometry Engine**: Uses true astronomical solar zenith angle ($\cos Z$) rather than clock time to switch dynamically between day and night processing modes.
* **Nocturnal Fog Detection**: Leverages the $3.9\,\mu\text{m}$ (MIR) vs. $10.8\,\mu\text{m}$ (TIR) brightness temperature difference ($\text{BTD}$) to detect low-level liquid water droplet clouds with negative emissivity signatures.
* **Low-Sun-Angle Transition Guard**: Smoothly blends model predictions with neighbor bounds during dawn/dusk twilight transitions when reflectance calibration is unstable.

### 5. Universal Geolocation-Locked Patch Cropping (`cropping_mechanism/`)
* **Coordinate Mapping**: Converts geodetic latitude/longitude coordinates to pixel space for both **INSAT Mercator projections** and **GOES ABI Fixed Grid Geostationary angles**.
* **Same-Grid Locking**: Locks pixel window boundaries across multiple spectral bands of different native resolutions, ensuring zero spatial drift between channels.
* **Self-Describing NetCDF Export**: Outputs cropped regions as standard CF-compliant `.nc` files with full spatial coordinate arrays and JSON manifests.

### 6. Quantitative Validation & Flow Visualization
* **Multi-Metric Suite**: Computes **PSNR**, **SSIM**, **RMSE (Kelvin)**, **MAE**, **FSIM**, and **LPIPS**.
* **Dense Optical Flow (HSV)**: Visualizes motion vector fields using Middlebury HSV color wheels with overlaid vector quiver arrows.
* **Intensity Hexbin Correlation Plots**: High-density scatter plots comparing interpolated vs. ground-truth pixel values against the theoretical ideal line ($y=x$).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Universal Data Ingestion & Calibration"]
        A1["INSAT-3D/3DR/3DS HDF5 (.h5)"] -->|Count-to-Kelvin LUT| B1["Calibrated BT13 & BT8"]
        A2["GOES-16/19 NetCDF (.nc)"] -->|Calibrated Radiances| B1
        B1 --> C1["BT Normalization: [190K, 310K]"]
        C1 --> C2["RGB False-Color Composite"]
        C1 --> C3["Raw Normalized TIR & WV Tensors"]
    end

    subgraph CROPPING["2. Geospatial Alignment & Cropping"]
        D1["Target Coordinates (Lat, Lon) & Patch Size"] --> E1["cropping_mechanism"]
        E1 -->|PyProj Geostationary Inversion| E2["Same-Grid Window Lock"]
        E2 --> F1["Padded Inputs (Multiples of 32)"]
    end

    subgraph MODEL["3. IFNet_GC Neural Pipeline"]
        F1 --> G1["Frozen Base Flownet (RIFE IFNet)"]
        G1 -->|Coarse-to-Fine Warping| H1["Bidirectional Flow Fields (4ch)"]
        G1 -->|Sigmoid Confidence| H2["Occlusion Mask (1ch)"]
        G1 -->|Warp & Blend| H3["Base Merged RGB (3ch)"]
        
        H1 & H2 & H3 & C3 --> I1["12-Channel Input Stem"]
        I1 --> J1["EvolutionRefinementNet (5 Residual Blocks, 64ch)"]
        J1 --> K1["Output Conv (head_rgb)"]
        K1 --> K2["Learned Gain Scaling (output_scale)"]
        
        H3 --> L1["Residual Fusion: Final = Base + Scale * Residual"]
        K2 --> L1
    end

    subgraph OUTPUTS["4. Analytics & Visualization"]
        L1 --> M1["Interpolated Frame (mid_frame.png)"]
        M1 --> N1["Cyclone Eye Tracker (Speed km/h, Heading)"]
        M1 --> N2["HSV Optical Flow Field + Vector Quivers"]
        M1 --> N3["Intensity Hexbin Scatter Plots & Metrics"]
        M1 --> N4["Dual Playback & Hierarchical GIFs"]
    end
```

---

## 🔬 Mathematical Formulations

### 1. Brightness Temperature Inversion & Normalization
For clean infrared window ($T_{\text{IR}}$) and water vapor ($T_{\text{WV}}$), raw Kelvin values are normalized using inverted bounds so cold cloud tops appear bright:
$$\tilde{I}_{\text{IR}} = \text{clip}\left(\frac{T_{\text{max}} - T}{T_{\text{max}} - T_{\text{min}}}, 0, 1\right), \quad T_{\text{min}}=190\,\text{K}, \; T_{\text{max}}=310\,\text{K}$$

### 2. Backward Optical Flow Warping
Warping source frame $I_0$ to target timestamp $t$ using estimated backward vector field $F_{t \to 0}$:
$$\widehat{I}_{t \leftarrow 0}(x) = I_0\left(x + F_{t \to 0}(x)\right)$$
Implemented via bilinear grid sampling with edge replication:
$$\widehat{I}_{\text{base}} = M \odot \widehat{I}_{t \leftarrow 0} + (1 - M) \odot \widehat{I}_{t \leftarrow 1}$$
where $M = \sigma(\text{mask})$ is the learned occlusion confidence map.

### 3. Residual Refinement Formulation
The 12-channel input tensor to `EvolutionRefinementNet`:
$$X_{\text{refine}} = \text{Concat}\left(\widehat{I}_{\text{base}}, \; F_{t \to 0}, \; F_{t \to 1}, \; M, \; \text{TIR}_0, \; \text{TIR}_1, \; \text{WV}_0, \; \text{WV}_1\right) \in \mathbb{R}^{B \times 12 \times H \times W}$$
$$\Delta I_{\text{residual}} = \mathbf{w}_{\text{scale}} \odot \text{Conv}_{\text{head}}\left(\text{ResBlocks}(X_{\text{refine}})\right)$$
$$I_{\text{final}} = \text{clip}\left(\widehat{I}_{\text{base}} + \Delta I_{\text{residual}}, \; 0, \; 1\right)$$

### 4. Multi-Objective Training Loss
$$\mathcal{L}_{\text{total}} = w_{\text{charb}} \mathcal{L}_{\text{charb}} + w_{\text{TIR}} \mathcal{L}_{\text{TIR}} + w_{\text{WV}} \mathcal{L}_{\text{WV}} + w_{\text{ssim}} \mathcal{L}_{\text{ssim}} + w_{\text{grad}} \mathcal{L}_{\text{grad}}$$
where:
* **Charbonnier Penalty**: $\mathcal{L}_{\text{charb}}(I, \widehat{I}) = \sqrt{\|I - \widehat{I}\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$
* **Edge / Gradient Loss**: $\mathcal{L}_{\text{grad}} = \|\nabla I - \nabla \widehat{I}\|_1$
* **SSIM Loss**: $\mathcal{L}_{\text{ssim}} = 1 - \text{SSIM}(I, \widehat{I})$

---

## 📁 Repository Directory Structure

```plaintext
IFNET_GC/
├── BAH-backend/                     # FastAPI Production Backend Server
│   ├── app/
│   │   ├── api/                     # Diagnostic & testing API endpoints
│   │   ├── core/                    # Preprocessing, storage, metrics, security
│   │   │   ├── hsv_visualizer.py    # Dense optical flow generator
│   │   │   ├── metrics.py           # PSNR, SSIM, RMSE, MAE, FSIM, LPIPS
│   │   │   ├── optical_flow.py      # Farneback flow utility
│   │   │   ├── preprocessing.py     # H5/NC extraction & RGB compositing
│   │   │   ├── security.py          # JWT authentication & bcrypt hashing
│   │   │   └── storage.py           # Supabase / cloud storage integration
│   │   ├── database/                # SQLAlchemy models & schemas
│   │   │   ├── database.py          # Session and DB engine init
│   │   │   ├── models.py            # Users, Experiments, Images, Metrics
│   │   │   └── schemas.py           # Pydantic validation schemas
│   │   ├── ml/                      # Machine learning inference engine
│   │   │   ├── evolution_model.py   # Frozen IFNet + EvolutionRefinementNet
│   │   │   ├── eye_tracker.py       # Cyclone eye detection & velocity estimator
│   │   │   ├── pysteps_run.py       # PySteps optical flow advection benchmark
│   │   │   ├── run_interpolation.py # Unified inference dispatcher
│   │   │   ├── training/            # Model training & fine-tuning scripts
│   │   │   ├── ui_overlays.py       # Tracking marker & radar overlays
│   │   │   └── universal_viewer.py  # Multi-spectral channel extraction
│   │   ├── routes/                  # API router definitions
│   │   │   ├── auth.py              # Login, register, profile
│   │   │   ├── crop.py              # Geospatial patch crop endpoint
│   │   │   ├── experiment.py        # Dataset ZIP preview, upload & run
│   │   │   └── universal.py         # Multi-spectral visualizer endpoint
│   │   └── main.py                  # FastAPI application entrypoint
│   ├── requirements.txt             # Backend Python dependencies
│   └── MIGRATION_GUIDE.md           # Architecture evolution documentation
│
├── bah-dashboard/                   # React 19 + TypeScript + Vite Dashboard
│   ├── public/                      # Static assets, blueprints, GLB models
│   │   ├── earth-globe-2.mp4        # Looping 3D Earth video asset
│   │   ├── satellite.glb            # INSAT-3DS 3D model
│   │   └── insat3ds_blueprint.png   # Spacecraft engineering blueprint
│   ├── src/
│   │   ├── components/              # Modular UI components
│   │   │   ├── landing/             # SpaceScene, 3D Earth, Satellite, Stars
│   │   │   └── ui/                  # Dashboard, GlassSurface, TargetCursor
│   │   ├── context/                 # AuthContext (JWT session state)
│   │   ├── pages/                   # Application views
│   │   │   ├── ArchitecturePage.tsx # Interactive pipeline architecture diagram
│   │   │   ├── AuthPage.tsx         # User authentication page
│   │   │   ├── ProfilePage.tsx      # User experiments history
│   │   │   └── ResultsPage.tsx      # Analysis, GIF players, charts, metrics
│   │   ├── store/                   # Client state management
│   │   ├── App.tsx                  # Routing & layout orchestration
│   │   └── index.css                # Global styles & Tailwind tokens
│   └── package.json                 # Frontend dependencies and scripts
│
├── cropping_mechanism/              # Universal Geolocation Cropping Tool
│   └── cropping_mechanism/
│       ├── core.py                  # Batch orchestration & NetCDF writer
│       ├── geo_utils.py             # Coordinate conversions & bounding boxes
│       ├── goes.py                  # GOES ABI projection decoder
│       └── insat.py                 # INSAT Mercator projection decoder
│
├── finalDeliverables.py             # Consecutive & recursive hierarchical interpolation
├── Floodscript.py                   # Multi-spectral flood tracking pipeline
├── fogtestingscript.py              # Solar zenith angle day/night fog engine
├── pysteps-run.py                   # Standalone PySteps benchmark script
├── hsv1.py / hsv2.py                # Dense HSV motion vector visualizers
├── satellite_model.py               # Satellite UNet baseline model
├── migrate.py                       # PostgreSQL schema migration utility
└── train_log/                       # Base RIFE HDv3 weights & modules
    ├── IFNet_HDv3.py                # Base network architecture
    ├── RIFE_HDv3.py                 # RIFE model wrapper
    ├── flownet.pkl                  # Pre-trained flow network weights
    ├── loss.py                      # Training loss functions
    └── warplayer.py                 # PyTorch backward warping operator
```

---

## 🚀 Installation & Setup

### Prerequisites
* **Operating System**: Linux (Ubuntu 20.04+) or macOS (Apple Silicon / Intel)
* **GPU**: NVIDIA GPU with CUDA 11.8+ recommended (CPU execution supported as fallback)
* **Python**: `3.10` or higher
* **Node.js**: `v18.0.0` or higher (with `npm` or `pnpm`)

---

### Backend Setup (`BAH-backend`)

1. **Navigate to the backend directory**:
   ```bash
   cd BAH-backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file in `BAH-backend/`:
   ```env
   # Database (PostgreSQL / Supabase / SQLite)
   DATABASE_URL=postgresql://postgres:password@localhost:5432/satellite_db
   # Or for local development without Postgres:
   # DATABASE_URL=sqlite:///./satellite.db

   # Supabase Storage Integration (Optional for local testing)
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-service-role-key
   SUPABASE_BUCKET=satellite-storage

   # JWT Authentication Secrets
   SECRET_KEY=your-super-secret-jwt-key
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=1440

   # Path to fine-tuned weights
   RIFE_GC_CHECKPOINT_PATH=/path/to/best_checkpoint.pth
   ```

5. **Initialize Database Tables**:
   ```bash
   python app/rebuild_db.py
   ```

6. **Launch the FastAPI Development Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   * Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
   * OpenAPI Specification: `http://localhost:8000/openapi.json`

---

### Frontend Setup (`bah-dashboard`)

1. **Navigate to the dashboard directory**:
   ```bash
   cd bah-dashboard
   ```

2. **Install node dependencies**:
   ```bash
   npm install
   ```

3. **Launch the Vite development server**:
   ```bash
   npm run dev
   ```
   * Open `http://localhost:5173` in your browser.

---

## 📡 API Reference Overview

| HTTP Method | Endpoint | Description | Payload / Parameters |
|:---|:---|:---|:---|
| `POST` | `/auth/signup` | Register new user account | JSON: `email`, `password`, `full_name` |
| `POST` | `/auth/login` | Authenticate & retrieve JWT | OAuth2 Form: `username`, `password` |
| `GET` | `/experiments/` | List user experiments | Query params, Auth Bearer token |
| `POST` | `/experiments/` | Create a new experiment | Query params: `model_name`, `model_version` |
| `POST` | `/experiments/preview-dataset` | Inspect ZIP dataset contents | File upload: `.zip` of `.nc`/`.h5` files |
| `POST` | `/experiments/target` | Select target frame from dataset | JSON: `target_time`, `frames` |
| `POST` | `/experiments/{id}/upload` | Upload explicit 4-file pair | Form: `t0_tir`, `t0_wv`, `t1_tir`, `t1_wv`, GT |
| `POST` | `/experiments/{id}/upload-zip` | Upload and extract dataset ZIP | Form: `dataset`, channel file names |
| `POST` | `/experiments/{id}/run` | Execute frame interpolation | Query: `model_type` (`ifnet-gc`, `pysteps`, `rife`, `linear`) |
| `POST` | `/universal/crop` | Geospatial square patch crop | Form: `file`, `lat`, `lon`, `patch_size` |
| `POST` | `/universal/channel-viewer` | Extract & colorize all channels | Form: `.nc` or `.h5` satellite file |

---

## 🛠️ Standalone CLI Utilities

### 1. Universal Geospatial Patch Cropper
Crop any mix of INSAT `.h5` or GOES `.nc` files to identical lat/lon square patches:
```bash
python cropping_mechanism/cropping_mechanism/core.py \
    /path/to/raw_satellite_data \
    --lat 19.5 \
    --lon 88.2 \
    --size 256 \
    --out ./cropped_output \
    --channels TIR1 WV
```

### 2. Hierarchical Full-Sequence Interpolation (`finalDeliverables.py`)
Generates smooth midpoint frames between consecutive real observations:
```bash
python finalDeliverables.py
# Follow the interactive prompt to choose recursion depth (1: 15 min, 2: 7.5 min, 3: 3.75 min)
```

### 3. Multi-Spectral Flood Tracking Pipeline (`Floodscript.py`)
Extracts multi-spectral water composites and builds multi-level inundation GIFs:
```bash
python Floodscript.py
```

### 4. Day/Night Fog Detection Engine (`fogtestingscript.py`)
Executes solar-zenith-angle guided fog interpolation with nocturnal BTD:
```bash
python fogtestingscript.py
```

### 5. Motion Vector HSV Generation (`hsv1.py` & `hsv2.py`)
Renders high-contrast optical flow fields with directional quiver arrows:
```bash
python hsv2.py --pred flow_pred.npy --gt flow_gt.npy --out ./visualizations/
```

---

## 📊 Benchmark & Evaluation Results

Evaluated across severe meteorological test cases (including **Hurricane Milton** and **Severe Cyclonic Storms** over the Bay of Bengal and Arabian Sea):

| Model Architecture | PSNR (dB) ↑ | SSIM ↑ | RMSE (K) ↓ | MAE ↓ | FSIM ↑ | LPIPS ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Blending** | 28.45 | 0.812 | 4.82 | 3.12 | 0.821 | 0.245 |
| **PySteps (Lucas-Kanade)** | 30.12 | 0.854 | 3.95 | 2.54 | 0.867 | 0.182 |
| **Stock RIFE HDv3** | 31.80 | 0.892 | 3.20 | 2.01 | 0.895 | 0.141 |
| **IFNet_GC (Ours - Base + Refine)** | **34.29** | **0.938** | **2.41** | **1.52** | **0.942** | **0.078** |

### Key Observations:
* **Convective Edge Sharpness**: Stock RIFE and PySteps introduce blurring at turbulent cloud boundaries; `EvolutionRefinementNet` preserves sharp eye-wall structure.
* **Thermodynamic Fidelity**: By directly consuming calibrated brightness temperatures, IFNet_GC avoids non-physical artifacts (e.g., phantom clouds appearing in warm dry regions).
* **Speed**: Inference executes in **$<45\,\text{ms}$ per $256 \times 256$ patch** on an NVIDIA RTX 3090/4090 GPU.

---

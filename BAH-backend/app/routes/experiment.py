import zipfile
import tempfile
import os
import shutil
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from PIL import Image
import io
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from app.database.database import get_db
from app.database import models
from app.core import security
from app.core.storage import upload_file_to_bucket
from app.core.metrics import compute_metrics
from app.core.preprocessing import build_rgb_composite, compute_gap_minutes, build_gap_map, extract_capture_time
from app.ml.run_interpolation import interpolate
from app.core.optical_flow import generate_optical_flow, ImageLoadError, InvalidImageError, FlowComputationError
import requests
import tempfile
import urllib.request
from app.ml.eye_tracker import detect_eye, calculate_motion
from app.ml.ui_overlays import draw_tracking_overlay
from app.ml.run_interpolation import load_band_pair, download_to_tempfile

router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"]
)


@router.get("/")
def get_experiments(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    experiments = db.query(models.Experiment).filter(models.Experiment.user_id == current_user.id).order_by(models.Experiment.created_at.desc()).all()
    return experiments

@router.post("/")
def create_experiment(
    model_name: str,
    model_version: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    experiment = models.Experiment(
        model_name=model_name,
        model_version=model_version,
        status="pending",
        user_id=current_user.id,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return {
        "experiment_id": experiment.id,
        "status": experiment.status,
    }



@router.post("/{experiment_id}/upload-zip")
def upload_input_images_zip(
    experiment_id: str,
    t0_tir_filename: str = Form(...),
    t0_wv_filename: str = Form(...),
    t1_tir_filename: str = Form(...),
    t1_wv_filename: str = Form(...),
    tmid_tir_filename: str | None = Form(None),
    tmid_wv_filename: str | None = Form(None),
    dataset: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    experiment = db.query(models.Experiment).filter(
        models.Experiment.id == experiment_id
    ).first()
    if experiment is None:
        raise HTTPException(404, "Experiment not found")

    try:
        file_bytes = dataset.file.read()
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as z:
            t0_tir_bytes = z.read(t0_tir_filename)
            t0_wv_bytes = z.read(t0_wv_filename)
            t1_tir_bytes = z.read(t1_tir_filename)
            t1_wv_bytes = z.read(t1_wv_filename)
            
            tmid_tir_bytes = None
            tmid_wv_bytes = None
            if tmid_tir_filename and tmid_wv_filename:
                tmid_tir_bytes = z.read(tmid_tir_filename)
                tmid_wv_bytes = z.read(tmid_wv_filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to extract files from ZIP: {e}")

    # ── Extract gap from file metadata (capture time, not upload time) ──
    try:
        gap_minutes = compute_gap_minutes(
            t0_tir_bytes, t0_tir_filename,
            t1_tir_bytes, t1_tir_filename,
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to compute gap from file metadata: {e}")

    experiment.gap_minutes = gap_minutes

    gap_map_bytes = build_gap_map(gap_minutes)
    gap_map_upload = upload_file_to_bucket(
        file_bytes=gap_map_bytes,
        original_filename="gap_map.png",
        folder=str(experiment_id),
    )
    experiment.gap_map_url = gap_map_upload["url"]

    composite_0 = build_rgb_composite(t0_tir_bytes, t0_tir_filename, t0_wv_bytes, t0_wv_filename)
    composite_1 = build_rgb_composite(t1_tir_bytes, t1_tir_filename, t1_wv_bytes, t1_wv_filename)

    composites = [
        (composite_0, t0_tir_filename, False, t0_tir_bytes, t0_wv_bytes, t0_wv_filename),
        (composite_1, t1_tir_filename, False, t1_tir_bytes, t1_wv_bytes, t1_wv_filename),
    ]
    
    if tmid_tir_bytes and tmid_wv_bytes:
        composite_mid = build_rgb_composite(tmid_tir_bytes, tmid_tir_filename, tmid_wv_bytes, tmid_wv_filename)
        composites.append((composite_mid, tmid_tir_filename, True, tmid_tir_bytes, tmid_wv_bytes, tmid_wv_filename))

    created = []
    for index, (composite_bytes, tir_filename, is_ground_truth, tir_bytes, wv_bytes, wv_filename) in enumerate(composites):
        filename = "ground_truth_mid.png" if is_ground_truth else f"composite_t{index}.png"
        upload = upload_file_to_bucket(
            file_bytes=composite_bytes,
            original_filename=filename,
            folder=str(experiment_id)
        )

        raw_tir_upload = upload_file_to_bucket(
            file_bytes=tir_bytes,
            original_filename=f"raw_tir_{index}_{os.path.basename(tir_filename)}",
            folder=str(experiment_id)
        )
        raw_wv_upload = upload_file_to_bucket(
            file_bytes=wv_bytes,
            original_filename=f"raw_wv_{index}_{os.path.basename(wv_filename)}",
            folder=str(experiment_id)
        )

        try:
            img = Image.open(io.BytesIO(composite_bytes))
            width, height = img.size
        except Exception:
            width = None
            height = None

        image = models.InputImage(
            experiment_id=experiment_id,
            sequence_no=index,
            image_url=upload["url"],
            filename=filename,
            width=width,
            height=height,
            file_size=len(composite_bytes),
            is_ground_truth=is_ground_truth,
            raw_tir_url=raw_tir_upload["url"],
            raw_wv_url=raw_wv_upload["url"],
        )
        db.add(image)
        created.append(image)

    db.commit()

    return {
        "message": "Satellite files processed and composites uploaded successfully",
        "gap_minutes": experiment.gap_minutes,
        "gap_map_url": experiment.gap_map_url,
        "t0_url": created[0].image_url if len(created) > 0 else None,
        "t1_url": created[1].image_url if len(created) > 1 else None,
        "ground_truth_mid_url": created[2].image_url if len(created) > 2 else None,
    }

@router.post("/{experiment_id}/upload")
def upload_input_images(
    experiment_id: str,
    t0_tir: UploadFile = File(...),
    t0_wv: UploadFile = File(...),
    t1_tir: UploadFile = File(...),
    t1_wv: UploadFile = File(...),
    tmid_tir: UploadFile | None = File(None),
    tmid_wv: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    experiment = db.query(models.Experiment).filter(
        models.Experiment.id == experiment_id
    ).first()
    if experiment is None:
        raise HTTPException(404, "Experiment not found")

    try:
        t0_tir_bytes = t0_tir.file.read()
        t0_wv_bytes = t0_wv.file.read()
        t1_tir_bytes = t1_tir.file.read()
        t1_wv_bytes = t1_wv.file.read()
        
        tmid_tir_bytes = None
        tmid_wv_bytes = None
        if tmid_tir and tmid_wv:
            tmid_tir_bytes = tmid_tir.file.read()
            tmid_wv_bytes = tmid_wv.file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded files: {e}")

    try:
        gap_minutes = compute_gap_minutes(
            t0_tir_bytes, t0_tir.filename,
            t1_tir_bytes, t1_tir.filename,
        )
    except Exception as e:
        raise HTTPException(400, f"Failed to compute gap from file metadata: {e}")

    experiment.gap_minutes = gap_minutes

    gap_map_bytes = build_gap_map(gap_minutes)
    gap_map_upload = upload_file_to_bucket(
        file_bytes=gap_map_bytes,
        original_filename="gap_map.png",
        folder=str(experiment_id),
    )
    experiment.gap_map_url = gap_map_upload["url"]
    db.commit()

    created = []
    
    pairs = [
        (t0_tir_bytes, t0_wv_bytes, t0_tir.filename, t0_wv.filename, False),
        (t1_tir_bytes, t1_wv_bytes, t1_tir.filename, t1_wv.filename, False),
    ]
    if tmid_tir_bytes and tmid_wv_bytes:
        pairs.append((tmid_tir_bytes, tmid_wv_bytes, tmid_tir.filename, tmid_wv.filename, True))

    for index, (tir_bytes, wv_bytes, tir_filename, wv_filename, is_ground_truth) in enumerate(pairs):
        try:
            composite_bytes = build_rgb_composite(tir_bytes, tir_filename, wv_bytes, wv_filename)
        except Exception as e:
            raise HTTPException(400, f"Failed to composite images for index {index}: {e}")

        filename = f"t{index}_composite.png"
        if is_ground_truth:
            filename = "ground_truth_mid.png"

        upload = upload_file_to_bucket(
            file_bytes=composite_bytes,
            original_filename=filename,
            folder=str(experiment_id)
        )
        raw_tir_upload = upload_file_to_bucket(
            file_bytes=tir_bytes,
            original_filename=f"raw_tir_{index}_{os.path.basename(tir_filename)}",
            folder=str(experiment_id)
        )
        raw_wv_upload = upload_file_to_bucket(
            file_bytes=wv_bytes,
            original_filename=f"raw_wv_{index}_{os.path.basename(wv_filename)}",
            folder=str(experiment_id)
        )

        try:
            img = Image.open(io.BytesIO(composite_bytes))
            width, height = img.size
        except Exception:
            width = None
            height = None

        image = models.InputImage(
            experiment_id=experiment_id,
            sequence_no=index,
            image_url=upload["url"],
            filename=filename,
            width=width,
            height=height,
            file_size=len(composite_bytes),
            is_ground_truth=is_ground_truth,
            raw_tir_url=raw_tir_upload["url"],
            raw_wv_url=raw_wv_upload["url"],
        )
        db.add(image)
        created.append(image)

    db.commit()

    return {
        "message": "Satellite files processed and composites uploaded successfully",
        "gap_minutes": experiment.gap_minutes,
        "gap_map_url": experiment.gap_map_url,
        "t0_url": created[0].image_url if len(created) > 0 else None,
        "t1_url": created[1].image_url if len(created) > 1 else None,
        "ground_truth_mid_url": created[2].image_url if len(created) > 2 else None,
    }

def _create_gif(frames_bytes: list[bytes], duration: int = 500) -> bytes:
    imgs = [Image.open(io.BytesIO(b)) for b in frames_bytes]
    buffer = io.BytesIO()
    imgs[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0
    )
    return buffer.getvalue()

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _create_difference_map(img1_bytes: bytes, img2_bytes: bytes) -> bytes:
    # img1 is interpolated (generated), img2 is ground truth (real)
    img1 = np.array(Image.open(io.BytesIO(img1_bytes)).convert("L"))
    img2 = np.array(Image.open(io.BytesIO(img2_bytes)).convert("L"))
    
    if img1.shape != img2.shape:
        target_h = min(img1.shape[0], img2.shape[0])
        target_w = min(img1.shape[1], img2.shape[1])
        img1 = np.array(Image.fromarray(img1).resize((target_w, target_h)))
        img2 = np.array(Image.fromarray(img2).resize((target_w, target_h)))
        
    flat_interp = img1.flatten()
    flat_real = img2.flatten()
    
    fig = Figure(figsize=(6, 6), dpi=100)
    canvas = FigureCanvas(fig)
    fig.patch.set_facecolor('#050505')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#050505')
    
    hb = ax.hexbin(flat_real, flat_interp, gridsize=50, cmap='inferno', mincnt=1)
    
    ax.plot([0, 255], [0, 255], color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (y=x)')
    
    ax.set_title("Real vs Interpolated Intensity", color='#e0e0e0', pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel("Ground Truth Intensity", color='#a0a0a0', fontsize=12, labelpad=10)
    ax.set_ylabel("Interpolated Intensity", color='#a0a0a0', fontsize=12, labelpad=10)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.tick_params(colors='#a0a0a0', labelsize=10)
    ax.grid(color='#ffffff', alpha=0.1, linestyle='--')
    
    # Hide spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Pixel Density', color='#a0a0a0', fontsize=12, labelpad=10)
    cb.ax.yaxis.set_tick_params(color='#a0a0a0', labelsize=10)
    for t in cb.ax.get_yticklabels():
        t.set_color('#a0a0a0')
    cb.outline.set_edgecolor('#333333')
    
    ax.legend(loc='upper left', frameon=False, labelcolor='#a0a0a0', fontsize=10)
    
    # Make axes perfectly square by setting exact identical proportional margins
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor=fig.get_facecolor(), transparent=False, dpi=150)
    return buffer.getvalue()

def _get_intensity_data(t0_bytes: bytes, tmid_real_bytes: bytes, tmid_interp_bytes: bytes, t1_bytes: bytes) -> dict:
    t0_img = np.array(Image.open(io.BytesIO(t0_bytes)).convert("L"))
    tmid_real_img = np.array(Image.open(io.BytesIO(tmid_real_bytes)).convert("L"))
    tmid_interp_img = np.array(Image.open(io.BytesIO(tmid_interp_bytes)).convert("L"))
    t1_img = np.array(Image.open(io.BytesIO(t1_bytes)).convert("L"))
    
    x = np.array([0, 1, 2])
    real_pts = np.array([float(t0_img.mean()), float(tmid_real_img.mean()), float(t1_img.mean())])
    interp_pts = np.array([float(t0_img.mean()), float(tmid_interp_img.mean()), float(t1_img.mean())])
    
    x_new = np.linspace(0, 2, 15)
    
    real_spline = make_interp_spline(x, real_pts, k=2)
    interp_spline = make_interp_spline(x, interp_pts, k=2)
    
    labels = []
    for val in x_new:
        if val == 0:
            labels.append("t0")
        elif val == 1:
            labels.append("t_mid")
        elif val == 2:
            labels.append("t1")
        else:
            labels.append("")
            
    return {
        "labels": labels,
        "real": [float(v) for v in real_spline(x_new)],
        "interpolated": [float(v) for v in interp_spline(x_new)]
    }

def _find_eye(img_bytes: bytes) -> tuple:
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))
    threshold = np.percentile(img, 95)
    y_coords, x_coords = np.where(img >= threshold)
    if len(y_coords) > 0:
        return (np.mean(x_coords), np.mean(y_coords))
    return (img.shape[1]/2, img.shape[0]/2)

def _get_cyclone_eye_data_from_tracker(eye0: tuple, eye_mid_real: tuple, eye_mid_interp: tuple, eye1: tuple) -> dict:
    x = np.array([0, 1, 2])
    real_x_pts = np.array([float(eye0[0]), float(eye_mid_real[0]), float(eye1[0])])
    real_y_pts = np.array([float(eye0[1]), float(eye_mid_real[1]), float(eye1[1])])
    interp_x_pts = np.array([float(eye0[0]), float(eye_mid_interp[0]), float(eye1[0])])
    interp_y_pts = np.array([float(eye0[1]), float(eye_mid_interp[1]), float(eye1[1])])
    
    x_new = np.linspace(0, 2, 15)
    
    labels = []
    for val in x_new:
        if val == 0:
            labels.append("t0")
        elif val == 1:
            labels.append("t_mid")
        elif val == 2:
            labels.append("t1")
        else:
            labels.append("")
            
    try:
        real_x = make_interp_spline(x, real_x_pts, k=2)(x_new)
        real_y = make_interp_spline(x, real_y_pts, k=2)(x_new)
        interp_x = make_interp_spline(x, interp_x_pts, k=2)(x_new)
        interp_y = make_interp_spline(x, interp_y_pts, k=2)(x_new)
    except Exception:
        real_x = np.interp(x_new, x, real_x_pts)
        real_y = np.interp(x_new, x, real_y_pts)
        interp_x = np.interp(x_new, x, interp_x_pts)
        interp_y = np.interp(x_new, x, interp_y_pts)
        
    return {
        "labels": labels,
        "real_x": [float(v) for v in real_x],
        "real_y": [float(v) for v in real_y],
        "interp_x": [float(v) for v in interp_x],
        "interp_y": [float(v) for v in interp_y]
    }

@router.post("/{experiment_id}/run")
def run_interpolation(
    experiment_id: str,
    model_type: str = "ifnet-gc",
    db: Session = Depends(get_db),
):
    experiment = db.query(models.Experiment).filter(
        models.Experiment.id == experiment_id
    ).first()
    if experiment is None:
        raise HTTPException(404, "Experiment not found")

    images = (
        db.query(models.InputImage)
        .filter(
            models.InputImage.experiment_id == experiment_id,
            models.InputImage.is_ground_truth == False,  # noqa: E712
        )
        .order_by(models.InputImage.sequence_no)
        .all()
    )
    if len(images) != 2:
        raise HTTPException(
            400,
            "Two input images are required. Upload via /upload first."
        )

    ground_truth_image = (
        db.query(models.InputImage)
        .filter(
            models.InputImage.experiment_id == experiment_id,
            models.InputImage.is_ground_truth == True,  # noqa: E712
        )
        .first()
    )

    if experiment.gap_minutes is None:
        raise HTTPException(
            400,
            "gap_minutes is missing for this experiment. Re-upload via /upload."
        )

    start_time = time.time()
    output_bytes = interpolate(
        str(images[0].image_url),
        str(images[1].image_url),
        str(images[0].raw_tir_url),
        str(images[1].raw_tir_url),
        str(images[0].raw_wv_url),
        str(images[1].raw_wv_url),
        gap_minutes=experiment.gap_minutes,
        model_type=model_type,
    )
    execution_time = time.time() - start_time

    # Compare the 2 composite input images to each other (input_comparisons).
    preprocessed0 = requests.get(str(images[0].image_url)).content
    preprocessed1 = requests.get(str(images[1].image_url)).content

    if model_type == "ifnet-gc":
        input_metrics = compute_metrics(preprocessed0, preprocessed1)
        input_comparison = models.InputComparison(
            experiment_id=experiment_id,
            comparison_type="input_vs_input",
            source_image_id=images[0].id,
            target_image_id=images[1].id,
            psnr=input_metrics["psnr"],
            ssim=input_metrics["ssim"],
            rmse=input_metrics["rmse"],
            mae=input_metrics["mae"],
            fsim=input_metrics["fsim"],
        )
        db.add(input_comparison)
        db.commit()

    # Upload generated image to Supabase.
    upload = upload_file_to_bucket(
        file_bytes=output_bytes,
        original_filename=f"mid_{model_type}.png",
        folder=str(experiment_id),
    )

    try:
        img = Image.open(io.BytesIO(output_bytes))
        width, height = img.size
    except Exception:
        width = None
        height = None

    hsv_flow_real_url = None
    hsv_flow_interpolated_url = None
    if ground_truth_image and model_type == "ifnet-gc":
        try:
            hsv_flow_real_bytes = generate_optical_flow(
                str(images[0].image_url),
                str(ground_truth_image.image_url)
            )
            upload_flow_real = upload_file_to_bucket(
                file_bytes=hsv_flow_real_bytes,
                original_filename="HSV_FLOW_REAL.png",
                folder=str(experiment_id)
            )
            hsv_flow_real_url = upload_flow_real["url"]

            hsv_flow_interpolated_bytes = generate_optical_flow(
                str(images[0].image_url),
                upload["url"]
            )
            upload_flow_interpolated = upload_file_to_bucket(
                file_bytes=hsv_flow_interpolated_bytes,
                original_filename="HSV_FLOW_INTERPOLATED.png",
                folder=str(experiment_id)
            )
            hsv_flow_interpolated_url = upload_flow_interpolated["url"]
        except (ImageLoadError, InvalidImageError, FlowComputationError) as e:
            print(f"Motion analysis failed: {e}")

    generated_image = models.GeneratedImage(
        experiment_id=experiment_id,
        image_url=upload["url"],
        execution_time=execution_time,
        width=width,
        height=height,
        file_size=len(output_bytes),
        hsv_flow_real_url=hsv_flow_real_url,
        hsv_flow_interpolated_url=hsv_flow_interpolated_url,
    )
    db.add(generated_image)

    experiment.status = "completed"
    experiment.execution_time = execution_time
    experiment.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(generated_image)

    # --- Eye Tracking Integration ---
    tracking_metrics = None
    annotated0, annotated_mid, annotated1 = preprocessed0, output_bytes, preprocessed1
    annotated_gt = None
    eye_mid_x_real = None
    eye_mid_y_real = None
    upload_interpolated_gif = {"url": None}

    if model_type == "ifnet-gc":
        print("Running Cyclone Eye Tracking...")
        tir0_path = download_to_tempfile(str(images[0].raw_tir_url))
        tir1_path = download_to_tempfile(str(images[1].raw_tir_url))
        
        try:
            bt13_0, _ = load_band_pair(tir0_path, tir0_path)
            bt13_1, _ = load_band_pair(tir1_path, tir1_path)
        except Exception as e:
            print(f"Error loading TIR for eye tracking: {e}")
            bt13_0, bt13_1 = None, None
            
        # Do not remove paths here yet, we need them for metadata extraction!
        
        if bt13_0 is not None and bt13_1 is not None:
            eye0_x, eye0_y, conf0 = detect_eye(bt13_0)
            eye1_x, eye1_y, conf1 = detect_eye(bt13_1, prev_x=eye0_x, prev_y=eye0_y)
            
            eye_mid_x = int((eye0_x + eye1_x) / 2)
            eye_mid_y = int((eye0_y + eye1_y) / 2)
            
            if ground_truth_image and ground_truth_image.raw_tir_url:
                gt_tir_path = download_to_tempfile(str(ground_truth_image.raw_tir_url))
                try:
                    bt13_gt, _ = load_band_pair(gt_tir_path, gt_tir_path)
                    eye_mid_x_real, eye_mid_y_real, _ = detect_eye(bt13_gt)
                except Exception:
                    pass
                finally:
                    if os.path.exists(gt_tir_path):
                        os.remove(gt_tir_path)
            
            from app.ml.eye_tracker import get_lat_lon_from_pixel
            latlon0 = get_lat_lon_from_pixel(tir0_path, eye0_x, eye0_y)
            latlon1 = get_lat_lon_from_pixel(tir1_path, eye1_x, eye1_y)
            
            motion = calculate_motion((eye0_x, eye0_y), (eye1_x, eye1_y), gap_hours=experiment.gap_minutes/60.0, latlon0=latlon0, latlon1=latlon1)
            
            tracking_metrics = {
                "speed_kmh": motion["speed_kmh"],
                "vx_kmh": motion["vx_kmh"],
                "vy_kmh": motion["vy_kmh"],
                "distance_km": motion["distance_km"],
                "direction_deg": motion["direction_deg"],
                "compass": motion["compass"],
                "confidence": conf0
            }
            
            speed_label = f"{motion['speed_kmh']} km/h {motion['compass']}"
            annotated0 = draw_tracking_overlay(preprocessed0, (eye0_x, eye0_y), [(eye0_x, eye0_y)], False, speed_label)
            annotated_mid = draw_tracking_overlay(output_bytes, (eye_mid_x, eye_mid_y), [(eye0_x, eye0_y)], True)
            annotated1 = draw_tracking_overlay(preprocessed1, (eye1_x, eye1_y), [(eye0_x, eye0_y), (eye1_x, eye1_y)], False)
            
            if ground_truth_image:
                ground_truth_bytes = requests.get(str(ground_truth_image.image_url)).content
                annotated_gt = draw_tracking_overlay(ground_truth_bytes, (eye_mid_x, eye_mid_y), [(eye0_x, eye0_y)], False)

        # Generate the interpolated GIF using annotated frames
        interpolated_gif_bytes = _create_gif([annotated0, annotated_mid, annotated1], duration=500)
        upload_interpolated_gif = upload_file_to_bucket(
            file_bytes=interpolated_gif_bytes,
            original_filename="interpolated_animation.gif",
            folder=str(experiment_id)
        )
        
        if os.path.exists(tir0_path):
            os.remove(tir0_path)
        if os.path.exists(tir1_path):
            os.remove(tir1_path)
    # Compare the generated (interpolated) image against the stored
    # ground-truth midpoint composite (uploaded as the 5th/6th inputs).
    if ground_truth_image and model_type == "ifnet-gc":
        ground_truth_bytes = requests.get(str(ground_truth_image.image_url)).content

        ground_truth_generated = models.GeneratedImage(
            experiment_id=experiment_id,
            image_url=ground_truth_image.image_url,
            width=ground_truth_image.width,
            height=ground_truth_image.height,
            file_size=ground_truth_image.file_size,
            is_ground_truth=True,
        )
        db.add(ground_truth_generated)
        db.commit()
        db.refresh(ground_truth_generated)

        gt_metrics = compute_metrics(output_bytes, ground_truth_bytes)
        output_comparison = models.OutputComparison(
            experiment_id=experiment_id,
            comparison_type="generated_vs_ground_truth",
            source_image_id=generated_image.id,
            target_image_id=ground_truth_generated.id,
            psnr=gt_metrics["psnr"],
            ssim=gt_metrics["ssim"],
            rmse=gt_metrics["rmse"],
            mae=gt_metrics["mae"],
            fsim=gt_metrics["fsim"],
            lpips=gt_metrics["lpips"],
        )
        db.add(output_comparison)
        db.commit()
        
        if annotated_gt is None:
            ground_truth_bytes = requests.get(str(ground_truth_image.image_url)).content
            annotated_gt = ground_truth_bytes

        real_gif_bytes = _create_gif([annotated0, annotated_gt, annotated1], duration=500)
        upload_real_gif = upload_file_to_bucket(
            file_bytes=real_gif_bytes,
            original_filename="real_animation.gif",
            folder=str(experiment_id)
        )
        real_gif_url = upload_real_gif["url"]
        
        diff_map_bytes = _create_difference_map(output_bytes, ground_truth_bytes)
        upload_diff_map = upload_file_to_bucket(
            file_bytes=diff_map_bytes,
            original_filename="difference_map.png",
            folder=str(experiment_id)
        )
        diff_map_url = upload_diff_map["url"]
        
        intensity_graph_data = _get_intensity_data(preprocessed0, ground_truth_bytes, output_bytes, preprocessed1)
        cyclone_eye_graph_data = None
        if tracking_metrics is not None:
            cyclone_eye_graph_data = _get_cyclone_eye_data_from_tracker(
                (eye0_x, eye0_y),
                (eye_mid_x_real, eye_mid_y_real) if eye_mid_x_real else (eye_mid_x, eye_mid_y),
                (eye_mid_x, eye_mid_y),
                (eye1_x, eye1_y)
            )
    else:
        gt_metrics = None
        real_gif_url = None
        diff_map_url = None
        intensity_graph_data = None
        cyclone_eye_graph_data = None

    return {
        "message": "Interpolation completed",
        "generated_image_url": generated_image.image_url,
        "ground_truth_image_url": ground_truth_image.image_url if ground_truth_image else None,
        "gap_minutes": experiment.gap_minutes,
        "ground_truth_metrics": gt_metrics,
        "interpolated_gif_url": upload_interpolated_gif["url"],
        "real_gif_url": real_gif_url,
        "difference_map_url": diff_map_url,
        "intensity_graph_data": intensity_graph_data,
        "cyclone_eye_graph_data": cyclone_eye_graph_data,
        "hsv_flow_real": hsv_flow_real_url if ground_truth_image else None,
        "hsv_flow_interpolated": hsv_flow_interpolated_url if ground_truth_image else None,
        "tracking_metrics": tracking_metrics,
    }

@router.post("/preview-dataset")
def upload_dataset(
    dataset: UploadFile = File(...),
):

    if not dataset.filename.lower().endswith('.zip'):
        raise HTTPException(400, "Dataset must be a ZIP file")

    dataset_bytes = dataset.file.read()
    
    nc_count = 0
    h5_count = 0
    total_files = 0
    
    # Store parsed frames
    # timestamp -> {"tir": bool, "wv": bool, "tir_filename": str, "wv_filename": str}
    frames = {}
    
    try:
        with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as z:
            for filename in z.namelist():
                if filename.startswith('__MACOSX') or filename.startswith('.'):
                    continue
                ext = filename.lower().split('.')[-1]
                if ext not in ('nc', 'h5', 'hdf5'):
                    continue
                
                total_files += 1
                
                with z.open(filename) as f:
                    file_bytes = f.read()
                
                try:
                    if ext == 'nc':
                        nc_count += 1
                        capture_time = extract_capture_time(file_bytes, filename)
                    else:
                        h5_count += 1
                        capture_time = extract_capture_time(file_bytes, filename)
                except Exception as e:
                    continue
                
                # Round to nearest minute
                ts_key = capture_time.replace(second=0, microsecond=0).isoformat()
                if ts_key not in frames:
                    frames[ts_key] = {"tir": False, "wv": False, "tir_filename": None, "wv_filename": None}
                
                # Heuristic to identify channel
                upper_name = filename.upper()
                if "TIR" in upper_name or "C13" in upper_name or "IR" in upper_name or "_B5" in upper_name or "_B6" in upper_name:
                    frames[ts_key]["tir"] = True
                    frames[ts_key]["tir_filename"] = filename
                if "WV" in upper_name or "C08" in upper_name or "_B3" in upper_name or "_B4" in upper_name:
                    frames[ts_key]["wv"] = True
                    frames[ts_key]["wv_filename"] = filename
                    
                if ext == 'nc' and not frames[ts_key]["tir"] and not frames[ts_key]["wv"]:
                    if "M6C13" in upper_name:
                        frames[ts_key]["tir"] = True
                        frames[ts_key]["tir_filename"] = filename
                    elif "M6C08" in upper_name:
                        frames[ts_key]["wv"] = True
                        frames[ts_key]["wv_filename"] = filename

    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid ZIP file")
        
    valid_frames_count = sum(1 for f in frames.values() if f["tir"] and f["wv"])
    
    return {
        "dataset_name": dataset.filename,
        "total_files": total_files,
        "nc_files": nc_count,
        "h5_files": h5_count,
        "valid_frames_count": valid_frames_count,
        "frames": frames
    }

@router.post("/target")
def select_target(
    payload: dict,
):
    target_time_str = payload.get("target_time")
    dataset_frames = payload.get("frames", {})
    if not target_time_str or not dataset_frames:
        raise HTTPException(400, "target_time and frames are required")
        
    try:
        target_time = datetime.fromisoformat(target_time_str.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "Invalid target time format")
        
    timestamps = []
    for ts_str in dataset_frames.keys():
        try:
            timestamps.append(datetime.fromisoformat(ts_str))
        except:
            pass
            
    timestamps.sort()
    
    if not timestamps:
        raise HTTPException(400, "No frames available")
        
    if target_time <= timestamps[0] or target_time >= timestamps[-1]:
        return {"error": "Target time is outside the dataset range."}
        
    # Find prev and next
    prev_ts = None
    next_ts = None
    
    for t in timestamps:
        if t < target_time:
            prev_ts = t
        elif t > target_time:
            next_ts = t
            break
            
    if prev_ts is None:
        return {"error": "Unable to locate a previous frame."}
    if next_ts is None:
        return {"error": "Unable to locate a next frame."}
        
    prev_key = prev_ts.isoformat()
    next_key = next_ts.isoformat()
    
    prev_channels = {"tir": dataset_frames[prev_key]["tir"], "wv": dataset_frames[prev_key]["wv"]}
    next_channels = {"tir": dataset_frames[next_key]["tir"], "wv": dataset_frames[next_key]["wv"]}
    
    return {
        "previous_timestamp": prev_key,
        "next_timestamp": next_key,
        "previous_channels": prev_channels,
        "next_channels": next_channels,
        "previous_files": {
            "tir": dataset_frames[prev_key]["tir_filename"],
            "wv": dataset_frames[prev_key]["wv_filename"],
        },
        "next_files": {
            "tir": dataset_frames[next_key]["tir_filename"],
            "wv": dataset_frames[next_key]["wv_filename"],
        },
        "target_files": {
            "tir": dataset_frames[target_time_str]["tir_filename"] if target_time_str in dataset_frames else None,
            "wv": dataset_frames[target_time_str]["wv_filename"] if target_time_str in dataset_frames else None,
        }
    }

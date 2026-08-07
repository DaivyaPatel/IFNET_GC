from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
from typing import List

from app.core.storage import upload_file_to_bucket
from app.ml.universal_viewer import process_file_in_memory

router = APIRouter(prefix="/universal", tags=["Universal Tools"])

@router.post("/channel-viewer")
def extract_channels(
    file: UploadFile = File(...)
):
    """
    Accepts an INSAT .nc or .h5 file, extracts all available image channels,
    normalizes them, applies the appropriate colormaps, uploads them to storage,
    and returns their URLs.
    """
    if not file.filename.lower().endswith(('.nc', '.h5', '.hdf5')):
        raise HTTPException(400, "File must be a .nc or .h5/.hdf5 file")

    results = []
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        # Run ML logic to extract RGBs in memory
        images_data = process_file_in_memory(tmp_path, file.filename)
        
        # Upload each generated image to Supabase
        for img in images_data:
            upload_res = upload_file_to_bucket(
                file_bytes=img["bytes"],
                original_filename=img["name"],
                folder="universal_viewer"
            )
            results.append({
                "channel_name": img["name"].replace(".png", ""),
                "url": upload_res["url"]
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "message": "Extraction successful",
        "images": results
    }

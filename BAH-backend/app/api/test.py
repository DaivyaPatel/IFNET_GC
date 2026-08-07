from typing import List
from fastapi import APIRouter, UploadFile, File

router = APIRouter(tags=["Test"])

@router.post("/upload-test")
async def upload_test(files: List[UploadFile] = File(...)):
    return {
        "count": len(files),
        "files": [
            {
                "filename": f.filename,
                "content_type": f.content_type,
            }
            for f in files
        ],
    }
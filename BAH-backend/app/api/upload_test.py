from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/upload-test", tags=["Upload Test"])


@router.post("/")
async def upload(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }
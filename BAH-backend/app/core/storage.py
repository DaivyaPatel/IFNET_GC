import os
import uuid
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # service key, not anon key, for server-side uploads
BUCKET_NAME = "isro-bah"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

import requests

def upload_file_to_bucket(file_bytes: bytes, original_filename: str, folder: str) -> dict:
    ext = original_filename.split(".")[-1] if "." in original_filename else "png"
    unique_name = f"{folder}/{uuid.uuid4()}.{ext}"

    # Use direct requests POST to avoid httpx WinError 10035 on Windows
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{unique_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": f"image/{ext}"
    }
    
    response = requests.post(url, headers=headers, data=file_bytes)
    response.raise_for_status()

    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{unique_name}"

    return {"url": public_url, "path": unique_name}
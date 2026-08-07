import os
import uuid
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv("c:\\Users\\daivy\\SatelliteDashboard\\BAH-backend\\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "isro-bah"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def test_upload():
    file_bytes = b"test file content"
    ext = "png"
    unique_name = f"test/{uuid.uuid4()}.{ext}"
    try:
        res = supabase.storage.from_(BUCKET_NAME).upload(
            file=file_bytes,
            path=unique_name,
            file_options={"content-type": f"image/{ext}"}
        )
        print("Success:", res)
    except Exception as e:
        print("Error:", e)

test_upload()

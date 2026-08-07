import os
import uuid
import requests
from dotenv import load_dotenv

load_dotenv("c:\\Users\\daivy\\SatelliteDashboard\\BAH-backend\\.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "isro-bah"

def test_upload():
    file_bytes = b"test file content"
    ext = "png"
    unique_name = f"test/{uuid.uuid4()}.{ext}"
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{unique_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": f"image/{ext}"
    }
    print("URL:", url)
    print("Headers:", headers)
    response = requests.post(url, headers=headers, data=file_bytes)
    print("Status code:", response.status_code)
    print("Response text:", response.text)

test_upload()

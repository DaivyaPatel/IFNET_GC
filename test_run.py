import requests
import numpy as np
import cv2

base_url = "http://127.0.0.1:8000"

print("Signing up/Logging in...")
try:
    res = requests.post(f"{base_url}/auth/signup", json={"email": "test@test.com", "password": "password", "full_name": "Test User"})
except:
    pass
res = requests.post(f"{base_url}/auth/login", data={"username": "test@test.com", "password": "password"})
token = res.json().get("access_token")
if not token:
    print("Login failed", res.text)
    exit(1)

headers = {"Authorization": f"Bearer {token}"}

print("Creating experiment...")
res = requests.post(f"{base_url}/experiments/?model_name=IFNET-GC&model_version=best_checkpoint", headers=headers)
exp_id = res.json()["experiment_id"]

print("Uploading...")
img = np.zeros((256, 256, 3), dtype=np.uint8)
success, encoded = cv2.imencode('.png', img)
dummy_bytes = encoded.tobytes()

files = {
    "t0_tir": ("dummy.png", dummy_bytes, "image/png"),
    "t0_wv": ("dummy.png", dummy_bytes, "image/png"),
    "t1_tir": ("dummy.png", dummy_bytes, "image/png"),
    "t1_wv": ("dummy.png", dummy_bytes, "image/png"),
}
res = requests.post(f"{base_url}/experiments/{exp_id}/upload", headers=headers, files=files)
if not res.ok:
    print("Failed to upload", res.text)
    exit(1)

print("Running interpolation...")
try:
    res = requests.post(f"{base_url}/experiments/{exp_id}/run", headers=headers)
    if not res.ok:
        print("Run failed", res.text)
    else:
        print("Success! Backend did not crash.")
except requests.exceptions.RequestException as e:
    print("Backend CRASHED during run:", e)

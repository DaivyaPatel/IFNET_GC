import sys
import os

# Add project root to path so 'app.ml...' imports work everywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.test import router as test_router
from app.api.upload_test import router as upload_test_router
from app.routes.experiment import router as experiment_router
from app.routes.auth import router as auth_router
from app.routes.universal import router as universal_router
from app.routes.crop import router as crop_router

app = FastAPI(title="Satellite Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables if they don't exist
from app.database import models
from app.database.database import engine
models.Base.metadata.create_all(bind=engine)

app.include_router(experiment_router)
app.include_router(auth_router)
app.include_router(universal_router)
app.include_router(crop_router)


@app.get("/")
def root():
    return {"message": "Backend running"}
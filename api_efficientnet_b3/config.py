"""Centralized environment configuration.

Loads `.env` for local dev (python-dotenv). On Cloud Run, env vars are
injected directly by the platform and load_dotenv() is a harmless no-op
since no .env file is present in the image.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Model ---
ONNX_PATH = os.environ.get("ONNX_PATH", str(BASE_DIR / "efficientnet_b3.onnx"))
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", str(BASE_DIR / "class_thresholds.csv"))

# --- GCS feedback storage (confirm/correct/new-produce) ---
FEEDBACK_BUCKET = os.environ.get("FEEDBACK_BUCKET", "vegdetect-pos-models")

# --- SSH tunnel + Postgres (tenant products DB) ---
# Set USE_SSH_TUNNEL=false when the app runs inside the same Azure network as
# the DB (VNet-integrated VM/Container App) and can reach DB_HOST directly.
USE_SSH_TUNNEL = os.environ.get("USE_SSH_TUNNEL", "true").lower() == "true"

SSH_HOST = os.environ.get("SSH_HOST")
SSH_PORT = int(os.environ.get("SSH_PORT", 22))
SSH_USER = os.environ.get("SSH_USER")
SSH_PKEY_PATH = os.environ.get("SSH_PKEY_PATH")

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# --- Azure Blob Storage (pending /api/detect-product images) ---
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "pending-detections")

# --- App ---
PORT = int(os.environ.get("PORT", 8080))

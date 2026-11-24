"""Paths, class labels, and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root: .../rw-visual-classifier
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CATALOG_PATH = DATA_DIR / "catalog.csv"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
ONNX_PATH = ARTIFACTS_DIR / "model.onnx"
BEST_CHECKPOINT = CHECKPOINTS_DIR / "best.pt"

# ReliefWeb API
RELIEFWEB_API_APPNAME = os.getenv("RELIEFWEB_API_APPNAME", "")
RELIEFWEB_API_URL = os.getenv(
    "RELIEFWEB_API_URL", "https://api.reliefweb.int/v2"
).rstrip("/")

# Content-format reference IDs
FORMAT_ID_MAP = 12
FORMAT_ID_INFOGRAPHIC = 12570

CLASS_NAMES = ("Map", "Infographic", "Other")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# Model / preprocessing
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dataset sampling
OTHER_SAMPLE_SIZE = 75_000
DEFAULT_LIMIT_PER_CLASS = 5_000
API_PAGE_LIMIT = 1000
DOWNLOAD_CONCURRENCY = 8
DOWNLOAD_RETRIES = 4

# Splits (by report id)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SPLIT_SEED = 42

# Training defaults
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 3

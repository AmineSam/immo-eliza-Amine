# config/paths.py
from pathlib import Path

# Project root = parent of this file's parent (config/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGE1_DIR = DATA_DIR / "stage1"
STAGE2_DIR = DATA_DIR / "stage2"
STAGE3_DIR = DATA_DIR / "stage3"
LOG_DIR = DATA_DIR / "logs"

# Files
RAW_FILE = RAW_DIR / "raw_dataset_v4.csv"
STAGE1_FILE = STAGE1_DIR / "cleaned_stage1.csv"
STAGE2_FILE = STAGE2_DIR / "cleaned_stage2.csv"
STAGE3_FILE = STAGE3_DIR / "cleaned_stage3.csv"

# Make sure dirs exist (idempotent)
for p in [RAW_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

"""
Central config — all paths and constants.
Import from here everywhere; never hardcode paths in routers.
"""
from pathlib import Path

# ── Path resolution ────────────────────────────────────────────────────────────
# Works whether Render runs from repo root or from backend/
BACKEND_DIR = Path(__file__).resolve().parent   # always = /repo/backend
REPO_ROOT   = BACKEND_DIR.parent                # always = /repo

MODELS_DIR  = REPO_ROOT / "Models"
DATA_DIR    = REPO_ROOT / "Data" / "Processed"

# ── Model file paths ───────────────────────────────────────────────────────────
MODEL_STUNTING    = MODELS_DIR / "random_forest_stunting.pkl"
MODEL_WASTING     = MODELS_DIR / "random_forest_wasting.pkl"
MODEL_UNDERWEIGHT = MODELS_DIR / "xgboost_underweight.pkl"

# ── Data file paths ────────────────────────────────────────────────────────────
DISTRICT_DATA_PATH    = DATA_DIR / "district_predictions_all_types.csv"
DISTRICT_MAPPING_PATH = DATA_DIR / "complete_district_mapping.csv"

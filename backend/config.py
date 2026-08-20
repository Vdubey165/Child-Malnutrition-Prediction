"""
Central config — all paths and constants.
Import from here everywhere; never hardcode paths in routers.
"""
from pathlib import Path

# ── Path resolution ────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent   # always = /repo/backend
REPO_ROOT   = BACKEND_DIR.parent                # always = /repo

MODELS_DIR  = REPO_ROOT / "Models"
DATA_DIR    = REPO_ROOT / "Data" / "Processed"
MODEL_DIR_V1 = MODELS_DIR / "v1"   # keep v1 pkls in their own subfolder                                     # don't mix with v2's json files
DISTRICT_ENHANCED_DATA_PATH = DATA_DIR / "district_malnutrition_enhanced.csv"
# ── Model file paths ───────────────────────────────────────────────────────────
# v2: child-level XGBoost classifiers (native JSON format), trained on NFHS-5
# Children's Recode microdata (206k-210k records) instead of 707 district
# aggregates. Predict per-child probability; aggregate across children if a
# district-level rate is needed.
MODEL_STUNTING    = MODELS_DIR / "final_model_stunting.json"
MODEL_WASTING     = MODELS_DIR / "final_model_wasting.json"
MODEL_UNDERWEIGHT = MODELS_DIR / "final_model_underweight.json"

# ── Data file paths ────────────────────────────────────────────────────────────
DISTRICT_DATA_PATH    = DATA_DIR / "district_predictions_all_types.csv"
DISTRICT_MAPPING_PATH = DATA_DIR / "complete_district_mapping.csv"

# ── State code mapping (NFHS-5 / DHS v024) ────────────────────────────────────
# Not defined here — this dict used to live in config.py but was never
# imported anywhere and had drifted out of sync with the version actually
# used at runtime. The live, correct mapping is
# services.district_mapping.STATE_MAPPING — import from there.
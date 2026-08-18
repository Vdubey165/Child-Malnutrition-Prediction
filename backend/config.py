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
# The model was trained with state as a raw numeric code; needed to translate
# a state name selected in the UI into the code the model expects.
STATE_CODES = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "NCT of Delhi", 8: "Rajasthan",
    9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh",
    13: "Nagaland", 14: "Manipur", 15: "Mizoram", 16: "Tripura",
    17: "Meghalaya", 18: "Assam", 19: "West Bengal", 20: "Jharkhand",
    21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Dadra & Nagar Haveli and Daman & Diu", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep",
    32: "Kerala", 33: "Tamil Nadu", 34: "Puducherry",
    35: "Andaman & Nicobar Islands", 36: "Telangana", 37: "Ladakh",
}

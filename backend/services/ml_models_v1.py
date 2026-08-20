"""
v1 district-aggregate model loading service — used ONLY by the scenario
simulator (/api/simulate), never by /api/predict.

Why a separate model from v2:
v2's XGBoost classifiers were trained on 232,920 INDIVIDUAL child records.
The simulator's input is a DISTRICT-AVERAGED feature vector. Feeding an
averaged vector into a model trained on individual rows is an
out-of-distribution input for that model (aggregation / ecological-inference
bias) — v1 was trained directly on 707 district-aggregate rows, so its
input distribution matches the simulator's input distribution exactly.

v1 models are plain pickled sklearn/XGBoost objects (Random Forest for
stunting/wasting, XGBoost regressor for underweight) — NOT the native-JSON
XGBoost classifiers v2 uses. Loaded with pickle.load(), not
xgb.XGBClassifier().load_model().
"""
import logging
import pickle

from config import MODEL_DIR_V1

logger = logging.getLogger(__name__)

# ── In-memory model store (separate from v2's _models in ml_models.py) ────────
_models_v1: dict = {}
models_v1_ready: bool = False
models_v1_error: str = ""

# Exact training order from Notebook/02_feature_engineering_and_modeling.ipynb
# feature_cols = [c for c in df.columns if c not in target_vars + id_vars]
# (knows_ors excluded — all-NaN column, dropped before training)
FEATURE_ORDER_V1 = [
    "wealth_index", "mother_edu_level", "mother_age", "mother_edu_years",
    "mother_bmi", "mother_works", "female_headed_hh", "child_age_months",
    "child_sex", "birth_interval", "birth_weight", "breastfeed_duration",
    "currently_breastfeed", "bcg_vaccination", "dpt_vaccination",
    "measles_vaccination",
]

REQUIRED_FILES = {
    "stunting":    "random_forest_stunting.pkl",
    "wasting":     "random_forest_wasting.pkl",
    "underweight": "xgboost_underweight.pkl",
}


def load_models_v1():
    """Load the three v1 pickled models. Does not raise — sets
    models_v1_ready/models_v1_error instead, so a missing v1 model degrades
    only /api/simulate, not the whole app (v2 /api/predict keeps working)."""
    global _models_v1, models_v1_ready, models_v1_error

    loaded = {}
    for key, filename in REQUIRED_FILES.items():
        path = MODEL_DIR_V1 / filename
        if not path.exists():
            msg = f"v1 model file not found: {path}"
            logger.warning(msg)
            models_v1_error = msg
            models_v1_ready = False
            return
        try:
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
            logger.info("Loaded v1 model: %s", filename)
        except Exception as e:
            msg = f"Failed to load {filename}: {e}"
            logger.error(msg)
            models_v1_error = msg
            models_v1_ready = False
            return

    _models_v1 = loaded
    models_v1_ready = True
    models_v1_error = ""
    logger.info("All v1 (district-aggregate) models loaded successfully.")


def get_models_v1() -> dict:
    return _models_v1
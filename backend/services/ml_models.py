"""
ML model loading service.
Models are loaded once at startup and held in memory.
If a model file is missing the app stays alive and returns 503
from prediction endpoints rather than crashing entirely.

v2: models are now child-level XGBoost classifiers saved in XGBoost's
native JSON format (not pickled sklearn/xgboost objects), so loading
uses xgb.XGBClassifier().load_model() instead of pickle.load().
"""
import logging
import xgboost as xgb

from config import MODEL_STUNTING, MODEL_WASTING, MODEL_UNDERWEIGHT

logger = logging.getLogger(__name__)

# ── In-memory model store ──────────────────────────────────────────────────────
_models: dict = {}
models_ready:  bool = False
models_error:  str  = ""


def load_models():
    """
    Load all three ML models from disk.
    Sets models_ready=True on success, models_error on failure.
    Does NOT raise — app stays alive either way.
    """
    global _models, models_ready, models_error

    required = {
        "stunting":    MODEL_STUNTING,
        "wasting":     MODEL_WASTING,
        "underweight": MODEL_UNDERWEIGHT,
    }

    loaded = {}
    for key, path in required.items():
        if not path.exists():
            msg = f"Model file not found: {path}"
            logger.error(msg)
            models_error = msg
            models_ready = False
            return
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            loaded[key] = model
            logger.info("Loaded model: %s", path.name)
        except Exception as e:
            msg = f"Failed to load {path.name}: {e}"
            logger.error(msg)
            models_error = msg
            models_ready = False
            return

    _models      = loaded
    models_ready = True
    models_error = ""
    logger.info("All models loaded successfully.")


def get_models() -> dict:
    return _models

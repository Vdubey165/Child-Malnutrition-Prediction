"""
ML model loading service.
Models are loaded once at startup and held in memory.
If a model file is missing the app stays alive and returns 503
from prediction endpoints rather than crashing entirely.
"""
import logging
import pickle

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
        "rf_stunting":     MODEL_STUNTING,
        "rf_wasting":      MODEL_WASTING,
        "xgb_underweight": MODEL_UNDERWEIGHT,
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
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
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

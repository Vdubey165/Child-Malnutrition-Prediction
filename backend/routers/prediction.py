"""
Prediction routes — /api/predict
"""
import logging
import numpy as np
from fastapi import APIRouter, HTTPException

from models.schemas import PredictionInput, PredictionOutput
from services import ml_models
from services.ml_models import get_models

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_risk(val: float, low: float, med: float) -> str:
    return "Low" if val < low else "Medium" if val < med else "High"


@router.post("/predict", response_model=PredictionOutput)
async def predict_malnutrition(input_data: PredictionInput):
    if not ml_models.models_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Models not available: {ml_models.models_error}",
        )

    features = np.array([[
        input_data.wealth_index,
        input_data.mother_edu_level,
        input_data.mother_age,
        input_data.mother_edu_years,
        input_data.mother_bmi,
        input_data.mother_works,
        input_data.female_headed_hh,
        input_data.child_age_months,
        input_data.child_sex,
        input_data.birth_interval,
        input_data.birth_weight,
        input_data.breastfeed_duration,
        input_data.currently_breastfeed,
        input_data.bcg_vaccination,
        input_data.dpt_vaccination,
        input_data.measles_vaccination,
    ]])

    try:
        ml = get_models()
        pred_stunting    = float(ml["rf_stunting"].predict(features)[0])
        pred_wasting     = float(ml["rf_wasting"].predict(features)[0])
        pred_underweight = float(ml["xgb_underweight"].predict(features)[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")

    return PredictionOutput(
        stunting=round(pred_stunting, 2),
        wasting=round(pred_wasting, 2),
        underweight=round(pred_underweight, 2),
        risk_level={
            "stunting":    _get_risk(pred_stunting,    20, 35),
            "wasting":     _get_risk(pred_wasting,     10, 20),
            "underweight": _get_risk(pred_underweight, 20, 35),
        },
    )
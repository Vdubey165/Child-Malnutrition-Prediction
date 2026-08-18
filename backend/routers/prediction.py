"""
Prediction routes — /api/predict

v2: models are child-level XGBoost classifiers. predict_proba() returns the
probability that THIS child is stunted/wasted/underweight; that probability
*100 is reported directly as the risk percentage for that child (this is no
longer a district-level rate — see PredictionInput docstring).
"""
import logging
import numpy as np
from fastapi import APIRouter, HTTPException

from models.schemas import PredictionInput, PredictionOutput
from services import ml_models
from services.ml_models import get_models

logger = logging.getLogger(__name__)
router = APIRouter()

# Training medians used to impute birth_interval for first-born children
# (b11 is undefined/missing for ~40% of children -- those with no preceding
# sibling -- so it cannot be a required field).
_BIRTH_INTERVAL_MEDIAN = 32.0


def _get_risk(val: float, low: float, med: float) -> str:
    return "Low" if val < low else "Medium" if val < med else "High"


# Feature order MUST match training exactly:
# v190, v106, v012, v133, v445, v714, v151, hw1, b4, b8, b11, m19, m4, h2, h3, h9, v394, v025, v024
def _build_features(input_data: PredictionInput) -> np.ndarray:
    birth_interval = input_data.birth_interval
    if birth_interval is None:
        birth_interval = _BIRTH_INTERVAL_MEDIAN

    return np.array([[
        input_data.wealth_index,
        input_data.mother_edu_level,
        input_data.mother_age,
        input_data.mother_edu_years,
        input_data.mother_bmi,
        input_data.mother_works,
        input_data.female_headed_hh,
        input_data.child_age_months,
        input_data.child_sex,
        input_data.child_age_years,
        birth_interval,
        input_data.birth_weight,
        input_data.breastfeed_duration,
        input_data.bcg_vaccination,
        input_data.dpt_vaccination,
        input_data.measles_vaccination,
        input_data.knows_ors,
        input_data.urban_rural,
        input_data.state,
    ]])


@router.post("/predict", response_model=PredictionOutput)
async def predict_malnutrition(input_data: PredictionInput):
    if not ml_models.models_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Models not available: {ml_models.models_error}",
        )

    features = _build_features(input_data)

    try:
        ml = get_models()
        pred_stunting    = float(ml["stunting"].predict_proba(features)[0][1])    * 100
        pred_wasting     = float(ml["wasting"].predict_proba(features)[0][1])     * 100
        pred_underweight = float(ml["underweight"].predict_proba(features)[0][1]) * 100
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

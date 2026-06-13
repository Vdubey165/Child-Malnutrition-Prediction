"""
FastAPI Backend for Child Malnutrition Prediction System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from district_mapping import enrich_district_data, get_state_name

app = FastAPI(title="Child Malnutrition Prediction API", version="1.0.0")

# CORS - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Robust path resolution ───────────────────────────────
# Works whether Render runs from repo root or from backend/
BACKEND_DIR = Path(__file__).resolve().parent  # always = /repo/backend
REPO_ROOT   = BACKEND_DIR.parent               # always = /repo

MODELS_DIR  = REPO_ROOT / "Models"
DATA_DIR    = REPO_ROOT / "Data" / "Processed"

# ── Data Models ──────────────────────────────────────────
class PredictionInput(BaseModel):
    wealth_index:        float = Field(..., ge=1,    le=5)
    mother_edu_level:    float = Field(..., ge=0,    le=3)
    mother_age:          float = Field(..., ge=15,   le=49)
    mother_edu_years:    float = Field(..., ge=0,    le=15)
    mother_bmi:          float = Field(..., ge=1000, le=4000)
    mother_works:        float = Field(..., ge=0,    le=1)
    female_headed_hh:    float = Field(..., ge=1,    le=2)
    child_age_months:    float = Field(..., ge=0,    le=59)
    child_sex:           float = Field(..., ge=1,    le=2)
    birth_interval:      float = Field(..., ge=1,    le=5)
    birth_weight:        float = Field(..., ge=400,  le=5000)
    breastfeed_duration: float = Field(..., ge=0,    le=90)
    currently_breastfeed:float = Field(..., ge=2000, le=8000)
    bcg_vaccination:     float = Field(..., ge=0,    le=2)
    dpt_vaccination:     float = Field(..., ge=0,    le=2)
    measles_vaccination: float = Field(..., ge=0,    le=3)

class PredictionOutput(BaseModel):
    stunting:    float
    wasting:     float
    underweight: float
    risk_level:  Dict[str, str]

# ── Model + data loading ─────────────────────────────────
ml_models     = {}
district_data = None

def load_models():
    global ml_models
    try:
        with open(MODELS_DIR / "random_forest_stunting.pkl",  "rb") as f:
            ml_models["rf_stunting"]    = pickle.load(f)
        with open(MODELS_DIR / "random_forest_wasting.pkl",   "rb") as f:
            ml_models["rf_wasting"]     = pickle.load(f)
        with open(MODELS_DIR / "xgboost_underweight.pkl",     "rb") as f:
            ml_models["xgb_underweight"]= pickle.load(f)
        print(f"✅ Models loaded from {MODELS_DIR}")
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        raise

def load_data():
    global district_data
    try:
        district_data = pd.read_csv(DATA_DIR / "district_predictions_all_types.csv")
        district_data = enrich_district_data(district_data)
        print(f"✅ Data loaded: {len(district_data)} districts from {DATA_DIR}")
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_models()
    load_data()

# ── Routes ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Child Malnutrition Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(ml_models) == 3,
        "districts_loaded": len(district_data) if district_data is not None else 0,
    }

@app.post("/api/predict", response_model=PredictionOutput)
async def predict_malnutrition(input_data: PredictionInput):
    features = np.array([[
        input_data.wealth_index,        input_data.mother_edu_level,
        input_data.mother_age,          input_data.mother_edu_years,
        input_data.mother_bmi,          input_data.mother_works,
        input_data.female_headed_hh,    input_data.child_age_months,
        input_data.child_sex,           input_data.birth_interval,
        input_data.birth_weight,        input_data.breastfeed_duration,
        input_data.currently_breastfeed,input_data.bcg_vaccination,
        input_data.dpt_vaccination,     input_data.measles_vaccination,
    ]])

    pred_stunting    = float(ml_models["rf_stunting"].predict(features)[0])
    pred_wasting     = float(ml_models["rf_wasting"].predict(features)[0])
    pred_underweight = float(ml_models["xgb_underweight"].predict(features)[0])

    def get_risk(val, low, med):
        return "Low" if val < low else "Medium" if val < med else "High"

    return PredictionOutput(
        stunting=round(pred_stunting, 2),
        wasting=round(pred_wasting, 2),
        underweight=round(pred_underweight, 2),
        risk_level={
            "stunting":    get_risk(pred_stunting,    20, 35),
            "wasting":     get_risk(pred_wasting,     10, 20),
            "underweight": get_risk(pred_underweight, 20, 35),
        }
    )

@app.get("/api/districts")
async def get_all_districts(limit: Optional[int] = 100):
    df = district_data.head(limit)
    return {"total": len(district_data), "districts": df.to_dict("records")}

@app.get("/api/districts/{district_id}")
async def get_district_by_id(district_id: int):
    district = district_data[district_data["district"] == district_id]
    if district.empty:
        raise HTTPException(status_code=404, detail="District not found")
    return district.iloc[0].to_dict()

@app.get("/api/statistics")
async def get_statistics():
    return {
        "national_average": {
            "stunting":    round(district_data["actual_stunting"].mean(),    2),
            "wasting":     round(district_data["actual_wasting"].mean(),     2),
            "underweight": round(district_data["actual_underweight"].mean(), 2),
        },
        "total_districts": len(district_data),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
Pydantic schemas for request bodies and response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class PredictionInput(BaseModel):
    wealth_index:         float = Field(..., ge=1,    le=5)
    mother_edu_level:     float = Field(..., ge=0,    le=3)
    mother_age:           float = Field(..., ge=15,   le=49)
    mother_edu_years:     float = Field(..., ge=0,    le=15)
    mother_bmi:           float = Field(..., ge=1000, le=4000)
    mother_works:         float = Field(..., ge=0,    le=1)
    female_headed_hh:     float = Field(..., ge=1,    le=2)
    child_age_months:     float = Field(..., ge=0,    le=59)
    child_sex:            float = Field(..., ge=1,    le=2)
    birth_interval:       float = Field(..., ge=1,    le=5)
    birth_weight:         float = Field(..., ge=400,  le=5000)
    breastfeed_duration:  float = Field(..., ge=0,    le=90)
    # Note: encoded survey response, range 2000–8000 is the NFHS-5 coded value
    currently_breastfeed: float = Field(..., ge=2000, le=8000)
    bcg_vaccination:      float = Field(..., ge=0,    le=2)
    dpt_vaccination:      float = Field(..., ge=0,    le=2)
    measles_vaccination:  float = Field(..., ge=0,    le=3)


class PredictionOutput(BaseModel):
    stunting:    float
    wasting:     float
    underweight: float
    risk_level:  Dict[str, str]

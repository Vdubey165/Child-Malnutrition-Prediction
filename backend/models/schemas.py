"""
Pydantic schemas for request bodies and response models.

v2: child-level input schema (one child's characteristics -> that child's
predicted risk), replacing the old district-aggregate-mean input. Two
fields from the original schema were built on mislabeled NFHS-5 variables
and have been corrected:
  - birth_weight: was reading v437 ("Respondent's weight in kilograms" —
    the MOTHER's weight, not birth weight). Now reads m19, the true
    "Birth weight in kilograms" variable, in grams.
  - currently_breastfeed: was reading m19 (child's true birth weight,
    mislabeled). Removed as a duplicate/nonsensical field. birth_interval
    now correctly reads b11 ("Preceding birth interval in months") instead
    of b8 ("Current age of child"), and child_age_years was added as its
    own correctly-labeled field for the value the old birth_interval field
    was actually populated with.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class PredictionInput(BaseModel):
    wealth_index:         float = Field(..., ge=1,    le=5,    description="Wealth index quintile (1=poorest, 5=richest)")
    mother_edu_level:     float = Field(..., ge=0,    le=3,    description="0=no education, 1=primary, 2=secondary, 3=higher")
    mother_age:           float = Field(..., ge=15,   le=49)
    mother_edu_years:     float = Field(..., ge=0,    le=20)
    mother_bmi:           float = Field(..., ge=1200, le=6000, description="BMI x100, e.g. 22.5 BMI = 2250")
    mother_works:         float = Field(..., ge=0,    le=1)
    female_headed_hh:     float = Field(..., ge=1,    le=2,    description="1=male head, 2=female head")
    child_age_months:     float = Field(..., ge=0,    le=59)
    child_sex:            float = Field(..., ge=1,    le=2,    description="1=male, 2=female")
    child_age_years:      float = Field(..., ge=0,    le=4)
    birth_interval:       Optional[float] = Field(None, ge=5, le=280, description="Months since preceding birth; leave blank/None for first-born children")
    birth_weight:         float = Field(..., ge=500,  le=5850, description="Birth weight in grams")
    breastfeed_duration:  float = Field(..., ge=0,    le=90,   description="Months breastfed")
    bcg_vaccination:      float = Field(..., ge=0,    le=3)
    dpt_vaccination:      float = Field(..., ge=0,    le=3)
    measles_vaccination:  float = Field(..., ge=0,    le=3)
    knows_ors:            float = Field(..., ge=0,    le=2,    description="Mother's knowledge of ORS packets: 0=no, 1=yes")
    urban_rural:          float = Field(..., ge=1,    le=2,    description="1=urban, 2=rural")
    state:                int   = Field(..., ge=1,    le=37,   description="NFHS-5 state code — see services.district_mapping.STATE_MAPPING")


class PredictionOutput(BaseModel):
    stunting:    float
    wasting:     float
    underweight: float
    risk_level:  Dict[str, str]
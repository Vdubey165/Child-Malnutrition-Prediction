# Integration Instructions

All files compile/smoke-tested clean. Drop these in and you're done — no
other files need touching.

## 1. Replace these files in your repo exactly as-is:
- `backend/config.py`
- `backend/services/ml_models.py`
- `backend/models/schemas.py`
- `backend/routers/prediction.py`
- `frontend/src/pages/Prediction.jsx`

## 2. Replace the model files:
- Copy `Models/final_model_stunting.json`, `final_model_wasting.json`,
  `final_model_underweight.json` into your repo's `Models/` folder
  (delete the old `.pkl` files — no longer used)

## 3. No changes needed to:
- `main.py` (already calls `load_models()` generically)
- `requirements.txt` (`xgboost>=2.0.0` was already listed)
- `Prediction.css` (new fields use the same `.form-group` classes)

## 4. Test locally before deploying:
```bash
cd backend
uvicorn main:app --reload
# then POST to /api/predict with a body matching the new PredictionInput schema
# (see backend/models/schemas.py for field names/ranges)
```

## What changed, in one line each
- Models: 707-row district regressors → child-level XGBoost classifiers
  (206k-210k NFHS-5 child records), aggregated via probability averaging
- Fixed a real bug: `birth_weight` was reading mother's weight (`v437`)
  instead of true birth weight (`m19`); `birth_interval` was reading
  child's current age (`b8`) instead of true birth interval (`b11`)
- Added `state` as a required input — it's now a top-3 predictor for
  wasting and underweight
- Removed the nonsensical `currently_breastfeed` field (was mislabeled
  birth weight); added `child_age_years`, `knows_ors`, `urban_rural`
- Frontend copy changed from "district-aggregate" framing to
  "individual child" framing throughout — this is now a per-child
  screening tool, not a district scenario simulator

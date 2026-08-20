"""
Scenario simulator — "what if district X raised feature Y by Z?"

Uses the v1 district-aggregate model deliberately (see services/ml_models_v1.py
for why). Estimates are labeled as sensitivity/association, never as a
guaranteed forecast — the underlying models have R² 0.43-0.69 on held-out
test districts (0.50 stunting, 0.43 wasting, 0.69 underweight), and shifting a
district-wide average isn't something a policy can directly execute, so the
output is a decision-support signal, not a causal prediction.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import pandas as pd

from config import DISTRICT_ENHANCED_DATA_PATH
from services import ml_models_v1
from services.ml_models_v1 import get_models_v1, FEATURE_ORDER_V1
from services.district_mapping import get_district_name, get_state_name

router = APIRouter()

# Loaded once, reused across requests — same district-aggregate CSV the v1
# model was trained on, so scenario baselines match training-data ranges.
_district_df: Optional[pd.DataFrame] = None


def _get_district_df() -> pd.DataFrame:
    global _district_df
    if _district_df is None:
        _district_df = pd.read_csv(DISTRICT_ENHANCED_DATA_PATH)
    return _district_df


class SimulateInput(BaseModel):
    district_id: int
    # Only the features you actually expose as sliders in the UI need to be
    # sent; anything omitted keeps the district's real average.
    feature_deltas: Dict[str, float] = Field(default_factory=dict)


class SimulateOutput(BaseModel):
    district_id: int
    district_name: str
    state_name: str
    baseline: Dict[str, float]
    scenario: Dict[str, float]
    delta: Dict[str, float]
    risk_level_baseline: Dict[str, str]
    risk_level_scenario: Dict[str, str]
    applied_deltas: Dict[str, float]
    clamped_features: Dict[str, str]  # feature -> reason, if a delta was capped
    # feature -> caution message, if the applied shift is large relative to how
    # much that feature actually varies across real districts. This is a cheap
    # single-feature heuristic (not a true joint out-of-distribution check —
    # correlated features like wealth/education/BMI aren't co-moved), but it
    # catches the most common way a scenario drifts unrealistically: pushing
    # one lever far further than real districts ever differ from each other.
    large_shift_features: Dict[str, str] = Field(default_factory=dict)
    model_version: str = "v1-district-aggregate"
    disclaimer: str = (
        "Estimated association based on historical district-level patterns "
        "(R\u00b2 0.43\u20130.69 on held-out test districts), not a causal forecast. "
        "District-wide averages are not directly controllable by policy."
    )


def _get_risk(val: float, low: float, med: float) -> str:
    return "Low" if val < low else "Medium" if val < med else "High"


@router.post("/simulate", response_model=SimulateOutput)
async def simulate_scenario(input_data: SimulateInput):
    if not ml_models_v1.models_v1_ready:
        raise HTTPException(
            status_code=503,
            detail="Scenario simulator unavailable: v1 models not loaded.",
        )

    df = _get_district_df()
    row = df[df["district"] == input_data.district_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="District not found")
    row = row.iloc[0]

    # Design intent (see module docstring) is one active lever at a time —
    # moving several district-average features independently compounds
    # aggregation/ecological-inference error, since the resulting feature
    # combination can drift out of the region the model was trained on even
    # when each individual value stays within its own observed range. The
    # frontend already enforces this by only ever sending one key, but that's
    # a UI convention, not a guarantee — enforce it here too so a direct API
    # call (or a future multi-slider UI) can't silently bypass it.
    nonzero_deltas = [f for f, d in input_data.feature_deltas.items() if d != 0]
    if len(nonzero_deltas) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only one feature may be changed per scenario to avoid "
                f"compounding aggregation error. Received: {nonzero_deltas}"
            ),
        )

    baseline_vec = {f: float(row[f]) for f in FEATURE_ORDER_V1}
    scenario_vec = dict(baseline_vec)
    applied, clamped, large_shift = {}, {}, {}

    # Threshold for the "large shift" caution below: 1.0 standard deviation
    # of the feature's spread across all 707 real districts. Chosen so the
    # flag is actually reachable near the top of each slider's range —
    # checked against real data: wealth_index and mother_edu_years only
    # reach 1.18 and 1.34 std-devs at their slider max, so a higher
    # threshold would never fire for them at all. It flags shifts bigger
    # than how much real districts differ from each other, without
    # requiring joint-distribution/correlation modeling.
    LARGE_SHIFT_STD_MULTIPLE = 1.0

    for feature, delta in input_data.feature_deltas.items():
        if feature not in FEATURE_ORDER_V1:
            raise HTTPException(status_code=400, detail=f"Unknown feature: {feature}")
        lo, hi = df[feature].min(), df[feature].max()
        new_val = baseline_vec[feature] + delta
        clamped_val = min(max(new_val, lo), hi)
        if clamped_val != new_val:
            clamped[feature] = f"clamped to observed district range [{lo:.2f}, {hi:.2f}]"
        scenario_vec[feature] = clamped_val
        applied_delta = clamped_val - baseline_vec[feature]
        applied[feature] = applied_delta

        std = df[feature].std()
        if std > 0:
            n_std = abs(applied_delta) / std
            if n_std > LARGE_SHIFT_STD_MULTIPLE:
                large_shift[feature] = (
                    f"this shift is {n_std:.1f}\u00d7 the feature's std-dev across "
                    "real districts \u2014 larger than how much most districts "
                    "actually differ from one another"
                )

    models = get_models_v1()
    X_base = pd.DataFrame([[baseline_vec[f] for f in FEATURE_ORDER_V1]], columns=FEATURE_ORDER_V1)
    X_scen = pd.DataFrame([[scenario_vec[f] for f in FEATURE_ORDER_V1]], columns=FEATURE_ORDER_V1)

    baseline_pred = {
        "stunting":    float(models["stunting"].predict(X_base)[0]),
        "wasting":     float(models["wasting"].predict(X_base)[0]),
        "underweight": float(models["underweight"].predict(X_base)[0]),
    }
    scenario_pred = {
        "stunting":    float(models["stunting"].predict(X_scen)[0]),
        "wasting":     float(models["wasting"].predict(X_scen)[0]),
        "underweight": float(models["underweight"].predict(X_scen)[0]),
    }
    delta_pred = {k: round(scenario_pred[k] - baseline_pred[k], 2) for k in baseline_pred}

    risk_bounds = {"stunting": (20, 35), "wasting": (10, 20), "underweight": (20, 35)}

    return SimulateOutput(
        district_id=input_data.district_id,
        district_name=get_district_name(input_data.district_id),
        state_name=get_state_name(int(row["state"])),
        baseline={k: round(v, 2) for k, v in baseline_pred.items()},
        scenario={k: round(v, 2) for k, v in scenario_pred.items()},
        delta=delta_pred,
        risk_level_baseline={k: _get_risk(baseline_pred[k], *risk_bounds[k]) for k in baseline_pred},
        risk_level_scenario={k: _get_risk(scenario_pred[k], *risk_bounds[k]) for k in scenario_pred},
        applied_deltas={k: round(v, 3) for k, v in applied.items()},
        clamped_features=clamped,
        large_shift_features=large_shift,
    )
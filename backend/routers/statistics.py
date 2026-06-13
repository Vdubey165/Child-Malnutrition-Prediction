"""
Statistics routes — /api/statistics
"""
import logging
from fastapi import APIRouter, HTTPException

from services.district_data import get_district_data, data_ready, data_error

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/statistics")
async def get_statistics():
    if not data_ready:
        raise HTTPException(
            status_code=503,
            detail=f"District data not available: {data_error}",
        )
    df = get_district_data()
    return {
        "national_average": {
            "stunting":    round(df["actual_stunting"].mean(),    2),
            "wasting":     round(df["actual_wasting"].mean(),     2),
            "underweight": round(df["actual_underweight"].mean(), 2),
        },
        "total_districts": len(df),
    }

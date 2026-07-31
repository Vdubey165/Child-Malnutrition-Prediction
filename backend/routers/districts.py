"""
District data routes — /api/districts/*
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from services import district_data
from services.district_data import get_district_data

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_data():
    """Raise 503 if district data isn't loaded yet."""
    if not district_data.data_ready:
        raise HTTPException(
            status_code=503,
            detail=f"District data not available: {district_data.data_error}",
        )


@router.get("/districts")
async def get_all_districts(
    limit:  int           = 100,
    offset: int           = 0,
    state:  Optional[str] = None,
):
    """
    Returns paginated district list.
    Optional ?state= filter by state_name.
    Optional ?offset= for pagination.
    """
    _require_data()
    df = get_district_data().copy()

    if state:
        df = df[df["state_name"].str.lower() == state.lower()]

    total   = len(df)
    paged   = df.iloc[offset : offset + limit]

    return {
        "total":     total,
        "offset":    offset,
        "limit":     limit,
        "districts": paged.to_dict("records"),
    }


@router.get("/districts/{district_id}")
async def get_district_by_id(district_id: int):
    _require_data()
    df       = get_district_data()
    district = df[df["district"] == district_id]
    if district.empty:
        raise HTTPException(status_code=404, detail="District not found")
    return district.iloc[0].to_dict()
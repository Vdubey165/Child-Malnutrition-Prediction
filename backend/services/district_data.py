"""
District data loading service.
Loads and holds the district predictions CSV in memory.
"""
import logging
import pandas as pd

from config import DISTRICT_DATA_PATH
from services.district_mapping import enrich_district_data

logger = logging.getLogger(__name__)

_district_data: pd.DataFrame | None = None
data_ready: bool = False
data_error: str  = ""


def load_district_data():
    """
    Load district predictions CSV and enrich with names.
    Does NOT raise — sets data_ready flag instead.
    """
    global _district_data, data_ready, data_error

    if not DISTRICT_DATA_PATH.exists():
        msg = f"Data file not found: {DISTRICT_DATA_PATH}"
        logger.error(msg)
        data_error = msg
        data_ready = False
        return

    try:
        df             = pd.read_csv(DISTRICT_DATA_PATH)
        _district_data = enrich_district_data(df)
        data_ready     = True
        data_error     = ""
        logger.info("District data loaded: %d districts from %s", len(_district_data), DISTRICT_DATA_PATH)
    except Exception as e:
        msg = f"Failed to load district data: {e}"
        logger.error(msg)
        data_error = msg
        data_ready = False


def get_district_data() -> pd.DataFrame | None:
    return _district_data

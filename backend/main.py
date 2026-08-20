"""
Child Malnutrition Prediction API — FastAPI entry point.
This file only wires things together. Logic lives in services/ and routers/.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.district_mapping import init_district_mapping
from services.ml_models import load_models
from services.ml_models_v1 import load_models_v1
from services.district_data import load_district_data
from routers import prediction, districts, statistics, simulate

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting up...")
    init_district_mapping()   # must run before load_district_data
    load_models()
    load_models_v1()          # non-fatal if missing — /api/simulate just returns 503
    load_district_data()
    logger.info("Startup complete.")
    yield
    logger.info("App shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Child Malnutrition Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # public read-only API — wildcard is fine here
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Child Malnutrition Prediction API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    from services.ml_models import models_ready
    from services.ml_models_v1 import models_v1_ready
    from services.district_data import data_ready, get_district_data
    df = get_district_data()
    return {
        "status":              "healthy" if models_ready and data_ready else "degraded",
        "models_loaded":       models_ready,
        "simulator_available": models_v1_ready,  # /api/simulate needs this
        "districts_loaded":    len(df) if df is not None else 0,
    }


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(prediction.router, prefix="/api")
app.include_router(districts.router,  prefix="/api")
app.include_router(statistics.router, prefix="/api")
app.include_router(simulate.router,   prefix="/api")


# ── Dev server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

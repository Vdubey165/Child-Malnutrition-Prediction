# Child Malnutrition Prediction API — Dockerfile for Google Cloud Run
#
# config.py resolves paths as:
#   BACKEND_DIR = this file's folder      -> /app/backend
#   REPO_ROOT   = BACKEND_DIR.parent      -> /app
#   MODELS_DIR  = REPO_ROOT / "Models"    -> /app/Models
#   DATA_DIR    = REPO_ROOT / "Data" / "Processed"  -> /app/Data/Processed
#
# So the image must preserve backend/, Models/, and Data/ as siblings under /app.
# Build this from the repo root (the folder containing backend/, Models/, Data/),
# not from inside backend/ itself.

FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy app code + the model/data files it loads at startup
COPY backend/ backend/
COPY Models/ Models/
COPY Data/ Data/

WORKDIR /app/backend

# Cloud Run injects $PORT at runtime (usually 8080).
# Shell form (via sh -c) is required so ${PORT} actually gets expanded.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

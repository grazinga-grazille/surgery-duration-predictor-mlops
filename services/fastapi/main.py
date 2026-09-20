"""FastAPI application entry point.

Exposes:
  GET  /          -> health check
  GET  /model     -> which MLflow/local artifacts are loaded
  POST /predict   -> surgery duration prediction

Loads the estimator + TF-IDF/SVD from MLflow MinIO artifacts when configured
(see MLFLOW_* env vars), otherwise falls back to local models/*.pkl.

Run from repo root:
  uv run uvicorn main:app --app-dir services/fastapi --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from predictor import predict
from schemas import PredictionRequest, PredictionResponse
from surgery_duration_predictor.serving import load_serving_artifacts

# Load repo-root .env when running uvicorn on the host (Compose injects env itself).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = FastAPI(
    title="Surgery Duration Predictor",
    description="Predicts OR surgery duration from procedure details.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_serving_artifacts()


@app.get("/")
def root():
    return {"status": "ok", "message": "Surgery Duration Predictor API is running"}


@app.get("/model")
def model_info():
    arts = load_serving_artifacts()
    return {
        "source": arts["source"],
        "model_family": arts["model_family"],
        "run_id": arts["run_id"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_duration(request: PredictionRequest):
    try:
        predicted_duration, model_family = predict(
            surgical_priority=request.surgical_priority,
            patient_type=request.patient_type,
            room=request.room,
            specialty=request.specialty,
            procedure_description=request.procedure_description,
        )
        return PredictionResponse(
            predicted_duration_minutes=predicted_duration,
            model_version=model_family,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

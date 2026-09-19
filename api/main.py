"""api/main.py — FastAPI application entry point.

Exposes:
  GET  /          -> health check
  POST /predict   -> surgery duration prediction

Run with:
  uv run uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException

from api.predictor import predict
from api.schemas import PredictionRequest, PredictionResponse
from surgery_duration_predictor.artifacts import load_artifacts

app = FastAPI(
    title="Surgery Duration Predictor",
    description="Predicts OR surgery duration from procedure details.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def root():
    return {"status": "ok", "message": "Surgery Duration Predictor API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_duration(request: PredictionRequest):
    try:
        predicted_duration = predict(
            surgical_priority=request.surgical_priority,
            patient_type=request.patient_type,
            room=request.room,
            specialty=request.specialty,
            procedure_description=request.procedure_description,
        )
        return PredictionResponse(predicted_duration_minutes=predicted_duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

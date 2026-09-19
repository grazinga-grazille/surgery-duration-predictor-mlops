"""api/predictor.py — Thin adapter between the FastAPI endpoint and the package.

Loads artifacts once at startup via the package's load_artifacts(), then
delegates prediction to surgery_duration_predictor.predict.predict().
"""

from surgery_duration_predictor.artifacts import load_artifacts
from surgery_duration_predictor.predict import predict as _predict


def predict(
    surgical_priority: int,
    patient_type: str,
    room: str,
    specialty: str,
    procedure_description: str,
) -> float:
    """Run the model and return predicted duration in minutes."""
    arts = load_artifacts()
    return _predict(
        arts["rf_model"],
        arts["tfidf"],
        arts["svd"],
        arts["feature_columns"],
        surgical_priority,
        patient_type,
        room,
        specialty,
        procedure_description,
    )

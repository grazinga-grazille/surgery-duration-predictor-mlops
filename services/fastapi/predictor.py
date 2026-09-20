"""services/fastapi/predictor.py — Thin adapter to the ML package."""

from surgery_duration_predictor.predict import predict as _predict
from surgery_duration_predictor.serving import load_serving_artifacts


def predict(
    surgical_priority: int,
    patient_type: str,
    room: str,
    specialty: str,
    procedure_description: str,
) -> tuple[float, str]:
    """Run the model and return (duration_minutes, model_family)."""
    arts = load_serving_artifacts()
    minutes = _predict(
        arts["model"],
        arts["tfidf"],
        arts["svd"],
        arts["feature_columns"],
        surgical_priority,
        patient_type,
        room,
        specialty,
        procedure_description,
    )
    return minutes, str(arts["model_family"])

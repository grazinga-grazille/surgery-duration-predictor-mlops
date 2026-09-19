"""api/schemas.py — Pydantic models for request validation and response serialization.

Defines the data contract for the /predict endpoint:
  - What the API accepts (PredictionRequest)
  - What the API returns (PredictionResponse)
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    surgical_priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Urgency score from 1 (most urgent) to 5 (elective)",
    )
    patient_type: str = Field(
        ...,
        description="Type of patient (e.g. Inpatient, Outpatient)",
    )
    room: str = Field(
        ...,
        description="Operating room identifier",
    )
    specialty: str = Field(
        ...,
        description="Procedure specialty (e.g. General Surgery)",
    )
    procedure_description: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the surgical procedure",
    )


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float = Field(
        ...,
        description="Predicted surgery duration in minutes",
    )
    model_version: str = Field(
        default="RF",
        description="Version of the model that generated this prediction",
    )

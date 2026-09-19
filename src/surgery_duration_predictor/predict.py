"""Inference logic.

Turns a fitted model + artifacts and an input row into a predicted duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surgery_duration_predictor.features import SVD_N_COMPONENTS, clean_text


def predict(
    model,
    tfidf,
    svd,
    feature_columns: list[str],
    surgical_priority: int,
    patient_type: str,
    room: str,
    specialty: str,
    procedure_description: str,
) -> float:
    """Return predicted surgery duration in minutes.

    Parameters
    ----------
    model, tfidf, svd, feature_columns : Artifacts from load_artifacts().
    surgical_priority : Urgency score 1–5.
    patient_type      : e.g. "Inpatient".
    room              : Operating room identifier.
    specialty         : Procedure specialty.
    procedure_description : Free-text procedure description.
    """
    cleaned = clean_text(procedure_description)
    tfidf_vec = tfidf.transform([cleaned])
    svd_vec = svd.transform(tfidf_vec)

    svd_df = pd.DataFrame(
        svd_vec, columns=[f"SVD_{i}" for i in range(SVD_N_COMPONENTS)]
    )
    X_num = pd.DataFrame({"SurgicalPriority": [surgical_priority]})
    X_cat_raw = pd.DataFrame({
        "PatientType": [patient_type],
        "Roomdescription": [room],
        "ProcedureSpecialtyDescription": [specialty],
    })
    X_cat = pd.get_dummies(X_cat_raw, drop_first=True, dtype=int)
    X = pd.concat([X_num, X_cat, svd_df], axis=1).reindex(
        columns=feature_columns, fill_value=0
    )
    return float(np.exp(model.predict(X)[0]))

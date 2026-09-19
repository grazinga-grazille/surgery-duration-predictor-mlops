"""Training logic.

train_model() takes the raw DataFrame, fits the full pipeline, and returns
a dict of all artifacts. Keeping it as a function (not a script) means
scripts, notebooks, tests, and pipelines can all call it without copying code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from surgery_duration_predictor.features import (
    BOOKED_COL,
    SPECIALTY_COL,
    TARGET_COL,
    TEXT_COL,
    build_features,
)


def train_model(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Fit the full pipeline and return a dict of all artifacts.

    Parameters
    ----------
    df           : Cleaned DataFrame from load_data().
    random_state : Seed for reproducibility.

    Returns
    -------
    dict with keys:
        rf_model, tfidf, svd, feature_columns, categorical_values,
        valid_combinations, procedure_examples, model_stats, test_results
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)

    # Fit feature pipeline on training data only
    X_train, y_train, tfidf, svd = build_features(train_df, fit=True)
    X_test, y_test, _, _ = build_features(test_df, tfidf=tfidf, svd=svd)

    feature_columns = X_train.columns.tolist()
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # Train Random Forest on log-transformed target
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate on test set
    preds_log = rf.predict(X_test)
    preds = np.exp(preds_log)
    actuals = np.exp(y_test.values)
    booked = test_df[BOOKED_COL].values

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    r2 = float(rf.score(X_test, y_test))
    booked_rmse = float(np.sqrt(mean_squared_error(actuals, booked)))
    booked_mae = float(mean_absolute_error(actuals, booked))

    model_stats = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "booked_rmse": booked_rmse,
        "booked_mae": booked_mae,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "y_mean": float(df[TARGET_COL].mean()),
        "y_median": float(df[TARGET_COL].median()),
        "y_std": float(df[TARGET_COL].std()),
    }

    test_results = pd.DataFrame({
        "actual": actuals,
        "predicted": preds,
        "residual": preds - actuals,
        "specialty": test_df[SPECIALTY_COL].values,
    })

    # ── Lookup tables for cascading sidebar dropdowns ─────────────────────────
    categorical_values = {
        "PatientType": sorted(df["PatientType"].dropna().unique().tolist()),
        "Roomdescription": sorted(df["Roomdescription"].dropna().unique().tolist()),
        "ProcedureSpecialtyDescription": sorted(
            df[SPECIALTY_COL].dropna().unique().tolist()
        ),
    }

    patient_to_specialty = (
        df.groupby("PatientType")[SPECIALTY_COL]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    patient_specialty_to_room = (
        df.groupby(["PatientType", SPECIALTY_COL])["Roomdescription"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    valid_combinations = {
        "patient_to_specialty": patient_to_specialty,
        "patient_specialty_to_room": patient_specialty_to_room,
    }

    procedure_examples = (
        df.groupby(SPECIALTY_COL)[TEXT_COL]
        .apply(lambda s: s.dropna().unique().tolist())
        .to_dict()
    )

    return {
        "rf_model": rf,
        "tfidf": tfidf,
        "svd": svd,
        "feature_columns": feature_columns,
        "categorical_values": categorical_values,
        "valid_combinations": valid_combinations,
        "procedure_examples": procedure_examples,
        "model_stats": model_stats,
        "test_results": test_results,
    }

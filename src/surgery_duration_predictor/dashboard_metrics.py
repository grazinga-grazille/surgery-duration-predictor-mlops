"""Build / load dashboard metrics aligned with the serving model.

ML Analysis charts and Business Analysis numbers should reflect the model
FastAPI serves (XGBoost from MLflow when configured), not the legacy RF
``model_stats.pkl`` from ``scripts/train.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from surgery_duration_predictor.artifacts import MODELS_DIR
from surgery_duration_predictor.business import (
    financial_impact_by_specialty,
    load_analysis_config,
    resource_utilization_by_specialty,
)
from surgery_duration_predictor.data import load_analysis_data, load_data
from surgery_duration_predictor.features import (
    BOOKED_COL,
    SPECIALTY_COL,
    TARGET_COL,
    build_features,
)
from surgery_duration_predictor.model_comparison import (
    evaluate_predictions,
    fit_xgboost,
    prepare_splits,
)
from surgery_duration_predictor.predict import predict as predict_one

DASHBOARD_MODEL_STATS = "dashboard_model_stats.pkl"
DASHBOARD_TEST_RESULTS = "dashboard_test_results.pkl"
DASHBOARD_FINANCIAL = "dashboard_financial.pkl"
DASHBOARD_RESOURCE = "dashboard_resource.pkl"
DASHBOARD_META = "dashboard_meta.pkl"


def _score_frame(df: pd.DataFrame, arts: dict[str, Any]) -> np.ndarray:
    """Score every row with serving transformers + model (vectorized)."""
    X, _, _, _ = build_features(df, tfidf=arts["tfidf"], svd=arts["svd"], fit=False)
    X = X.reindex(columns=arts["feature_columns"], fill_value=0)
    preds_log = arts["model"].predict(X)
    return np.exp(np.asarray(preds_log, dtype=float))


def evaluate_artifacts_on_holdout(
    arts: dict[str, Any],
    df: pd.DataFrame | None = None,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Hold-out metrics using the *serving* feature transformers (not re-fit).

    Uses the same train/test row split as ``prepare_splits`` / compare_models
    (seed 42, 80/20) so numbers line up with the XGBoost compare run.
    """
    if df is None:
        df = load_data()
    _, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    X_test, y_test, _, _ = build_features(
        test_df, tfidf=arts["tfidf"], svd=arts["svd"], fit=False
    )
    X_test = X_test.reindex(columns=arts["feature_columns"], fill_value=0)
    preds_log = arts["model"].predict(X_test)
    stats, results = evaluate_predictions(
        y_test,
        preds_log,
        test_df[BOOKED_COL].values,
        specialty=test_df[SPECIALTY_COL].values,
    )
    stats = dict(stats)
    stats["train_size"] = len(df) - len(test_df)
    stats["test_size"] = len(test_df)
    stats["y_mean"] = float(df[TARGET_COL].mean())
    stats["y_median"] = float(df[TARGET_COL].median())
    stats["y_std"] = float(df[TARGET_COL].std())
    stats["model_family"] = arts.get("model_family", "unknown")
    stats["source"] = arts.get("source", "unknown")
    stats["run_id"] = arts.get("run_id")
    return stats, results


def build_xgboost_serving_bundle(
    df: pd.DataFrame | None = None,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit Optuna-best XGBoost on the compare split (offline fallback)."""
    if df is None:
        df = load_data()
    split = prepare_splits(df, random_state=random_state)
    fitted = fit_xgboost(split["X_train"], split["y_train"], split["X_test"])
    return {
        "model": fitted["model"],
        "tfidf": split["tfidf"],
        "svd": split["svd"],
        "feature_columns": split["feature_columns"],
        "model_family": "xgboost",
        "run_id": None,
        "source": "local_rebuild",
    }


def resolve_serving_bundle(*, prefer_mlflow: bool = True) -> dict[str, Any]:
    """Load MLflow serving artifacts when configured; else rebuild XGBoost."""
    if prefer_mlflow:
        try:
            from surgery_duration_predictor.serving import load_serving_artifacts

            arts = load_serving_artifacts()
            # Local RF fallback is not the dashboard target — rebuild XGB instead.
            if arts.get("source") == "mlflow" or arts.get("model_family") == "xgboost":
                return arts
        except Exception:
            pass
    return build_xgboost_serving_bundle()


def compute_business_summaries(
    arts: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score the analysis date window and return financial + resource tables."""
    cfg = load_analysis_config(config_path)
    analysis = cfg["analysis"]
    cost = cfg["cost"]

    df = load_analysis_data(
        date_start=analysis["date_start"],
        date_end=analysis["date_end"],
    )
    predicted = _score_frame(df, arts)
    financial = financial_impact_by_specialty(
        df,
        predicted,
        undertime_rate=float(cost["undertime_rate"]),
        overtime_multiplier=float(cost["overtime_multiplier"]),
    )
    resource = resource_utilization_by_specialty(df)
    meta = {
        "date_start": analysis["date_start"],
        "date_end": analysis["date_end"],
        "n_cases": int(len(df)),
        "undertime_rate": float(cost["undertime_rate"]),
        "overtime_multiplier": float(cost["overtime_multiplier"]),
        "model_family": arts.get("model_family"),
        "source": arts.get("source"),
        "run_id": arts.get("run_id"),
        "total_baseline_cost": float(financial["baseline_cost"].sum()),
        "total_predicted_cost": float(financial["predicted_cost"].sum()),
        "total_savings": float(financial["savings"].sum()),
        "total_additional": int(resource["total_additional_possible"].sum()),
    }
    return financial, resource, meta


def save_dashboard_artifacts(
    *,
    output_dir: Path | None = None,
    prefer_mlflow: bool = True,
) -> dict[str, Any]:
    """Evaluate serving model, compute business tables, write pickles under models/."""
    out = output_dir or MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    arts = resolve_serving_bundle(prefer_mlflow=prefer_mlflow)
    stats, results = evaluate_artifacts_on_holdout(arts)
    financial, resource, meta = compute_business_summaries(arts)
    meta.update({
        "r2": stats["r2"],
        "rmse": stats["rmse"],
        "mae": stats["mae"],
        "test_size": stats["test_size"],
    })

    payloads = {
        DASHBOARD_MODEL_STATS: stats,
        DASHBOARD_TEST_RESULTS: results,
        DASHBOARD_FINANCIAL: financial,
        DASHBOARD_RESOURCE: resource,
        DASHBOARD_META: meta,
    }
    for name, obj in payloads.items():
        with open(out / name, "wb") as f:
            pickle.dump(obj, f)

    return {
        "model_stats": stats,
        "test_results": results,
        "financial": financial,
        "resource": resource,
        "meta": meta,
        "output_dir": str(out),
    }


def load_dashboard_artifacts(models_dir: Path | None = None) -> dict[str, Any] | None:
    """Load precomputed dashboard pickles, or None if any file is missing."""
    root = models_dir or MODELS_DIR
    needed = [
        DASHBOARD_MODEL_STATS,
        DASHBOARD_TEST_RESULTS,
        DASHBOARD_FINANCIAL,
        DASHBOARD_RESOURCE,
        DASHBOARD_META,
    ]
    if not all((root / name).exists() for name in needed):
        return None

    def _load(name: str):
        with open(root / name, "rb") as f:
            return pickle.load(f)

    return {
        "model_stats": _load(DASHBOARD_MODEL_STATS),
        "test_results": _load(DASHBOARD_TEST_RESULTS),
        "financial": _load(DASHBOARD_FINANCIAL),
        "resource": _load(DASHBOARD_RESOURCE),
        "meta": _load(DASHBOARD_META),
    }


# Re-export for callers that score one row the same way as the API
__all__ = [
    "build_xgboost_serving_bundle",
    "compute_business_summaries",
    "evaluate_artifacts_on_holdout",
    "load_dashboard_artifacts",
    "predict_one",
    "resolve_serving_bundle",
    "save_dashboard_artifacts",
]

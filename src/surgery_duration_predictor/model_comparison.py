"""Multi-model training helpers (Linear Regression, Random Forest, XGBoost).

Production serving still uses Random Forest via train_model(). These helpers
share the same feature pipeline so models can be compared fairly.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from surgery_duration_predictor.diagnostics import (
    format_assumption_report,
    validate_linear_assumptions,
)
from surgery_duration_predictor.features import (
    BOOKED_COL,
    SPECIALTY_COL,
    TARGET_COL,
    build_features,
)
from surgery_duration_predictor.mlflow_logging import log_compare_run

ModelName = Literal["linear", "random_forest", "xgboost"]


def prepare_splits(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Build train/test feature matrices with a shared feature pipeline."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    X_train, y_train, tfidf, svd = build_features(train_df, fit=True)
    X_test, y_test, _, _ = build_features(test_df, tfidf=tfidf, svd=svd)
    feature_columns = X_train.columns.tolist()
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "tfidf": tfidf,
        "svd": svd,
        "feature_columns": feature_columns,
        "booked_test": test_df[BOOKED_COL].values,
        "specialty_test": test_df[SPECIALTY_COL].values,
    }


def _metrics(actuals: np.ndarray, preds: np.ndarray, booked: np.ndarray, y_test_log, preds_log) -> dict:
    return {
        "r2_log": float(r2_score(y_test_log, preds_log)),
        "r2": float(r2_score(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "mae": float(mean_absolute_error(actuals, preds)),
        "booked_rmse": float(np.sqrt(mean_squared_error(actuals, booked))),
        "booked_mae": float(mean_absolute_error(actuals, booked)),
    }


def evaluate_predictions(
    y_test_log: pd.Series | np.ndarray,
    preds_log: np.ndarray,
    booked: np.ndarray,
    specialty: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Convert log-scale predictions to minutes and compute metrics."""
    preds = np.exp(preds_log)
    actuals = np.exp(np.asarray(y_test_log, dtype=float))
    stats = _metrics(actuals, preds, booked, y_test_log, preds_log)
    results = pd.DataFrame({
        "actual": actuals,
        "predicted": preds,
        "residual": preds - actuals,
    })
    if specialty is not None:
        results["specialty"] = specialty
    return stats, results


def fit_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    """Fit OLS on scaled features; optionally validate classical assumptions."""
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    train_pred = model.predict(X_train_s)
    test_pred = model.predict(X_test_s)

    diagnostics = None
    diagnostics_text = None
    if run_diagnostics:
        diagnostics = validate_linear_assumptions(X_train_s, y_train, train_pred)
        diagnostics_text = format_assumption_report(diagnostics)

    return {
        "model": model,
        "scaler": scaler,
        "preds_log": test_pred,
        "train_preds_log": train_pred,
        "diagnostics": diagnostics,
        "diagnostics_text": diagnostics_text,
    }


# Best Optuna params from models/tuning_xgboost.json (70 trials).
BEST_XGBOOST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.012645540454443964,
    "max_depth": 10,
    "n_estimators": 646,
    "subsample": 0.9013719848538555,
    "colsample_bytree": 0.6393325808934611,
    "min_child_weight": 4,
    "reg_lambda": 1.7794500569750102,
    "reg_alpha": 0.003047567485839047,
}

# Best Optuna params from models/tuning_random_forest.json (70 trials).
BEST_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 157,
    "max_depth": 21,
    "min_samples_leaf": 3,
    "min_samples_split": 20,
    "max_features": 0.3,
}


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit XGBoost with Optuna best params on the shared log-duration target."""
    model = XGBRegressor(
        **BEST_XGBOOST_PARAMS,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "preds_log": model.predict(X_test),
        "train_preds_log": model.predict(X_train),
    }


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit Random Forest with Optuna best params for side-by-side comparison."""
    model = RandomForestRegressor(
        **BEST_RF_PARAMS,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return {
        "model": model,
        "preds_log": model.predict(X_test),
        "train_preds_log": model.predict(X_train),
    }


def compare_models(
    df: pd.DataFrame,
    *,
    models: tuple[ModelName, ...] = ("linear", "xgboost", "random_forest"),
    random_state: int = 42,
    include_rf: bool = True,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Train selected models on one split and return metrics + LR diagnostics.

    By default trains Linear Regression (with assumption checks) and XGBoost.
    Random Forest is included for a like-for-like comparison with production.
    """
    if not include_rf:
        models = tuple(m for m in models if m != "random_forest")

    split = prepare_splits(df, random_state=random_state)
    X_train, y_train = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    booked, specialty = split["booked_test"], split["specialty_test"]

    fitted: dict[str, Any] = {}
    summary_rows = []
    mlflow_run_ids: dict[str, str] = {}

    default_params: dict[str, dict] = {
        "linear": {"model": "LinearRegression", "scaled_features": True},
        "xgboost": {**BEST_XGBOOST_PARAMS, "random_state": random_state},
        "random_forest": {**BEST_RF_PARAMS, "random_state": random_state, "n_jobs": -1},
    }

    for name in models:
        if name == "linear":
            print("Fitting Linear Regression + validating OLS assumptions ...")
            out = fit_linear_regression(X_train, y_train, X_test, run_diagnostics=True)
            if out["diagnostics_text"]:
                print(out["diagnostics_text"])
                print()
        elif name == "xgboost":
            print("Fitting XGBoost ...")
            out = fit_xgboost(X_train, y_train, X_test, random_state=random_state)
        elif name == "random_forest":
            print("Fitting Random Forest ...")
            out = fit_random_forest(X_train, y_train, X_test, random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {name}")

        stats, results = evaluate_predictions(
            y_test, out["preds_log"], booked, specialty
        )
        fitted[name] = {
            **out,
            "metrics": stats,
            "test_results": results,
            "y_mean": float(df[TARGET_COL].mean()),
            "y_median": float(df[TARGET_COL].median()),
            "y_std": float(df[TARGET_COL].std()),
            "train_size": len(split["train_df"]),
            "test_size": len(split["test_df"]),
        }
        summary_rows.append({"model": name, **stats})

        if log_mlflow:
            run_id = log_compare_run(
                model_family=name,
                metrics=stats,
                params={**default_params.get(name, {}), "random_state": random_state},
                n_train=len(split["train_df"]),
                n_test=len(split["test_df"]),
                diagnostics=out.get("diagnostics"),
                diagnostics_text=out.get("diagnostics_text"),
                model=out.get("model"),
                tfidf=split["tfidf"],
                svd=split["svd"],
                feature_columns=split["feature_columns"],
            )
            if run_id:
                mlflow_run_ids[name] = run_id

    summary = pd.DataFrame(summary_rows)
    return {
        "split": split,
        "models": fitted,
        "summary": summary,
        "mlflow_run_ids": mlflow_run_ids,
    }

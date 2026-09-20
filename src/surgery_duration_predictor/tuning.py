"""Optuna hyperparameter tuning for Random Forest and XGBoost.

Mirrors the forecasting-rental-bike-count notebook pattern:
  - hold out 20% test
  - carve 20% of the remaining block as validation (→ 64/16/20)
  - minimize validation MAE (minutes scale)
  - TPESampler + MedianPruner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from surgery_duration_predictor.features import BOOKED_COL, build_features

ModelKind = Literal["random_forest", "xgboost"]

DEFAULT_TUNING_CFG: dict[str, Any] = {
    "n_trials": 70,
    "train_fraction": 0.8,
    "valid_within_train": 0.2,
    "early_stopping_rounds": 30,
    "random_seed": 42,
}

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models"


def load_tuning_config(path: Path | None = None) -> dict[str, Any]:
    """Load tuning section from config.yaml, falling back to defaults."""
    cfg = dict(DEFAULT_TUNING_CFG)
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        cfg.update(raw.get("tuning", {}))
    return cfg


def prepare_tuning_splits(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    valid_within_train: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Random 64/16/20 split with features fit on the train fold only."""
    train_valid_df, test_df = train_test_split(
        df, test_size=1.0 - train_fraction, random_state=random_seed
    )
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_within_train,
        random_state=random_seed,
    )

    X_train, y_train, tfidf, svd = build_features(train_df, fit=True)
    feature_columns = X_train.columns.tolist()

    X_valid, y_valid, _, _ = build_features(valid_df, tfidf=tfidf, svd=svd)
    X_test, y_test, _, _ = build_features(test_df, tfidf=tfidf, svd=svd)
    X_valid = X_valid.reindex(columns=feature_columns, fill_value=0)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # Combined train+valid for final refit after tuning
    train_valid_df = pd.concat([train_df, valid_df], axis=0)
    X_tv, y_tv, tfidf_tv, svd_tv = build_features(train_valid_df, fit=True)
    feature_columns_tv = X_tv.columns.tolist()
    X_test_final, y_test_final, _, _ = build_features(
        test_df, tfidf=tfidf_tv, svd=svd_tv
    )
    X_test_final = X_test_final.reindex(columns=feature_columns_tv, fill_value=0)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_valid": X_tv,
        "y_train_valid": y_tv,
        "X_test_final": X_test_final,
        "y_test_final": y_test_final,
        "feature_columns": feature_columns,
        "feature_columns_final": feature_columns_tv,
        "tfidf": tfidf,
        "svd": svd,
        "tfidf_final": tfidf_tv,
        "svd_final": svd_tv,
        "booked_valid": valid_df[BOOKED_COL].values,
        "booked_test": test_df[BOOKED_COL].values,
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
    }


def _mae_minutes(y_log: np.ndarray | pd.Series, pred_log: np.ndarray) -> float:
    return float(mean_absolute_error(np.exp(np.asarray(y_log)), np.exp(pred_log)))


def _test_metrics(
    y_log: np.ndarray | pd.Series,
    pred_log: np.ndarray,
    booked: np.ndarray,
) -> dict[str, float]:
    actuals = np.exp(np.asarray(y_log, dtype=float))
    preds = np.exp(pred_log)
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "r2": float(r2_score(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "mae": float(mean_absolute_error(actuals, preds)),
        "booked_rmse": float(np.sqrt(mean_squared_error(actuals, booked))),
        "booked_mae": float(mean_absolute_error(actuals, booked)),
    }


def _rf_objective(trial: optuna.Trial, split: dict[str, Any], seed: int) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]
        ),
        "random_state": seed,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**params)
    model.fit(split["X_train"], split["y_train"])
    pred = model.predict(split["X_valid"])
    mae = _mae_minutes(split["y_valid"], pred)

    trial.report(mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return mae


def _xgb_objective(
    trial: optuna.Trial,
    split: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": early_stopping_rounds,
    }
    model = XGBRegressor(**params)
    model.fit(
        split["X_train"],
        split["y_train"],
        eval_set=[(split["X_valid"], split["y_valid"])],
        verbose=False,
    )
    pred = model.predict(split["X_valid"])
    mae = _mae_minutes(split["y_valid"], pred)

    trial.report(mae, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return mae


def _build_best_model(
    kind: ModelKind,
    best_params: dict[str, Any],
    seed: int,
    early_stopping_rounds: int,
) -> Any:
    clean = {k: v for k, v in best_params.items()}
    if kind == "random_forest":
        return RandomForestRegressor(
            **clean, random_state=seed, n_jobs=-1
        )
    return XGBRegressor(
        **clean,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=early_stopping_rounds,
    )


def tune_model(
    df: pd.DataFrame,
    kind: ModelKind,
    *,
    n_trials: int | None = None,
    tuning_cfg: dict[str, Any] | None = None,
    show_progress: bool = True,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Run an Optuna study for one model family and evaluate on held-out test."""
    cfg = dict(DEFAULT_TUNING_CFG)
    if tuning_cfg:
        cfg.update(tuning_cfg)
    if n_trials is not None:
        cfg["n_trials"] = n_trials

    seed = int(cfg["random_seed"])
    split = prepare_tuning_splits(
        df,
        train_fraction=float(cfg["train_fraction"]),
        valid_within_train=float(cfg["valid_within_train"]),
        random_seed=seed,
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{kind}_surgery_duration",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    early_stopping = int(cfg["early_stopping_rounds"])

    from surgery_duration_predictor.mlflow_logging import (
        EXPERIMENT_TUNE,
        _base_tags,
        log_tune_best,
        log_tune_trial,
        setup_tracking,
        start_run,
    )

    use_mlflow = log_mlflow and setup_tracking()

    def objective(trial: optuna.Trial) -> float:
        try:
            if kind == "random_forest":
                mae = _rf_objective(trial, split, seed)
            else:
                mae = _xgb_objective(trial, split, seed, early_stopping)
            pruned = False
        except optuna.TrialPruned:
            # Still log the reported value if available
            if use_mlflow and trial.params:
                intermediate = trial.intermediate_values
                mae_logged = float(intermediate.get(0, float("nan")))
                if mae_logged == mae_logged:  # not NaN
                    log_tune_trial(
                        model_family=kind,
                        trial_number=trial.number,
                        params=dict(trial.params),
                        valid_mae=mae_logged,
                        pruned=True,
                    )
            raise

        if use_mlflow:
            log_tune_trial(
                model_family=kind,
                trial_number=trial.number,
                params=dict(trial.params),
                valid_mae=float(mae),
                pruned=pruned,
            )
        return mae

    mlflow_run_id: str | None = None

    def _run_study_and_finalize() -> dict[str, Any]:
        nonlocal mlflow_run_id
        study.optimize(
            objective,
            n_trials=int(cfg["n_trials"]),
            show_progress_bar=show_progress,
        )

        best_params = dict(study.best_params)

        if kind == "xgboost":
            final_params = dict(best_params)
            model = XGBRegressor(
                **final_params,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
            model.fit(split["X_train_valid"], split["y_train_valid"])
        else:
            model = _build_best_model(kind, best_params, seed, early_stopping)
            model.fit(split["X_train_valid"], split["y_train_valid"])

        pred_test = model.predict(split["X_test_final"])
        test_metrics = _test_metrics(
            split["y_test_final"], pred_test, split["booked_test"]
        )

        result = {
            "kind": kind,
            "study": study,
            "best_params": best_params,
            "best_valid_mae": float(study.best_value),
            "test_metrics": test_metrics,
            "n_train": split["n_train"],
            "n_valid": split["n_valid"],
            "n_test": split["n_test"],
            "model": model,
            "config": cfg,
        }
        return result

    if use_mlflow:
        with start_run(
            experiment=EXPERIMENT_TUNE,
            run_name=f"tune_{kind}",
            tags=_base_tags(model_family=kind, stage="tune_best"),
        ) as parent:
            result = _run_study_and_finalize()
            # Persist JSON then attach to the parent run
            json_path = save_tuning_result(result)
            mlflow_run_id = log_tune_best(
                model_family=kind,
                best_params=result["best_params"],
                best_valid_mae=result["best_valid_mae"],
                test_metrics=result["test_metrics"],
                config=result["config"],
                n_train=result["n_train"],
                n_valid=result["n_valid"],
                n_test=result["n_test"],
                tuning_json_path=json_path,
                model=result["model"],
            )
            if parent is not None:
                mlflow_run_id = parent.info.run_id
            result["mlflow_run_id"] = mlflow_run_id
            result["tuning_json_path"] = json_path
            return result

    result = _run_study_and_finalize()
    result["mlflow_run_id"] = None
    result["tuning_json_path"] = save_tuning_result(result)
    return result


def save_tuning_result(result: dict[str, Any], output_dir: Path | None = None) -> Path:
    """Persist best params + metrics as JSON (study object is not serialised)."""
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tuning_{result['kind']}.json"
    payload = {
        "kind": result["kind"],
        "best_params": result["best_params"],
        "best_valid_mae": result["best_valid_mae"],
        "test_metrics": result["test_metrics"],
        "n_train": result["n_train"],
        "n_valid": result["n_valid"],
        "n_test": result["n_test"],
        "config": result["config"],
        "n_trials_completed": len(result["study"].trials),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def format_tuning_summary(result: dict[str, Any]) -> str:
    m = result["test_metrics"]
    rmse_imp = (1 - m["rmse"] / m["booked_rmse"]) * 100
    mae_imp = (1 - m["mae"] / m["booked_mae"]) * 100
    lines = [
        f"── {result['kind']} ───────────────────────────────────────",
        f"  split: train={result['n_train']:,}  valid={result['n_valid']:,}  "
        f"test={result['n_test']:,}",
        f"  best valid MAE: {result['best_valid_mae']:.2f} min",
        f"  best params:    {result['best_params']}",
        f"  test R²(log)={m['r2_log']:.4f}  R²(min)={m['r2']:.4f}",
        f"  test RMSE={m['rmse']:.1f}  MAE={m['mae']:.1f}",
        f"  vs booked: RMSE {rmse_imp:+.1f}%  MAE {mae_imp:+.1f}%",
    ]
    return "\n".join(lines)

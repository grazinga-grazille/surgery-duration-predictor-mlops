"""MLflow tracking helpers for compare / tune / champion runs.

Enabled when MLFLOW_TRACKING_URI is set (see .env.example).
If unset, helpers no-op so local scripts still run offline.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from surgery_duration_predictor.data import ML_CUTOFF_DATE

EXPERIMENT_COMPARE = "surgery-duration-compare"
EXPERIMENT_TUNE = "surgery-duration-tune"


def mlflow_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI", "").strip())


def setup_tracking() -> bool:
    """Configure MLflow tracking + MinIO/S3 artifact access. Returns True if enabled."""
    if not mlflow_enabled():
        return False

    import mlflow

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"].strip()
    mlflow.set_tracking_uri(tracking_uri)

    # MinIO / S3-compatible artifact store (used when server stores s3://… roots)
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "").strip()
    if endpoint:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", endpoint)
    # boto3 keys: prefer explicit AWS_* then fall back to MinIO root creds
    if not os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("MINIO_ROOT_USER"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["MINIO_ROOT_USER"]
    if not os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("MINIO_ROOT_PASSWORD"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MINIO_ROOT_PASSWORD"]
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    return True


def _ensure_experiment(name: str) -> None:
    import mlflow

    mlflow.set_experiment(name)


def _improvement_metrics(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    booked_rmse = metrics.get("booked_rmse")
    booked_mae = metrics.get("booked_mae")
    if booked_rmse and booked_rmse > 0:
        out["rmse_improvement_pct"] = (1 - metrics["rmse"] / booked_rmse) * 100
    if booked_mae and booked_mae > 0:
        out["mae_improvement_pct"] = (1 - metrics["mae"] / booked_mae) * 100
    return out


def _base_tags(*, model_family: str, stage: str) -> dict[str, str]:
    return {
        "model_family": model_family,
        "stage": stage,
        "ml_cutoff_date": str(ML_CUTOFF_DATE.date()),
        "project": "surgery-duration-predictor",
    }


def _log_params(params: dict[str, Any]) -> None:
    import mlflow

    clean = {}
    for k, v in params.items():
        if v is None:
            continue
        clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    if clean:
        mlflow.log_params(clean)


def _log_metrics(metrics: dict[str, Any]) -> None:
    import mlflow

    for k, v in metrics.items():
        if v is None:
            continue
        try:
            mlflow.log_metric(k, float(v))
        except (TypeError, ValueError):
            continue


@contextmanager
def start_run(
    *,
    experiment: str,
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Context manager; yields the active MLflow run or None if disabled."""
    if not setup_tracking():
        yield None
        return

    import mlflow

    _ensure_experiment(experiment)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def _log_model_artifact(model_family: str, model: Any) -> None:
    """Upload a fitted model under artifact_path='model' (MinIO / S3).

    Uses save_model + log_artifacts instead of log_model so we stay compatible
    with MLflow tracking servers that do not implement /logged-models (2.x).
    """
    import mlflow

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "model"
            if model_family == "xgboost":
                import mlflow.xgboost

                mlflow.xgboost.save_model(model, str(local_dir), model_format="ubj")
            else:
                # linear / random_forest / other sklearn estimators
                mlflow.sklearn.save_model(model, str(local_dir))
            mlflow.log_artifacts(str(local_dir), artifact_path="model")
    except Exception as exc:
        mlflow.log_param("model_log_error", str(exc)[:200])


def _log_feature_artifacts(
    *,
    tfidf: Any,
    svd: Any,
    feature_columns: list[str],
) -> None:
    """Upload TF-IDF / SVD / column list needed for online inference."""
    import pickle

    import mlflow

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, obj in (
                ("tfidf.pkl", tfidf),
                ("svd.pkl", svd),
                ("feature_columns.pkl", feature_columns),
            ):
                path = root / name
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
                mlflow.log_artifact(str(path), artifact_path="features")
    except Exception as exc:
        mlflow.log_param("features_log_error", str(exc)[:200])


def log_compare_run(
    *,
    model_family: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    n_train: int,
    n_test: int,
    diagnostics: dict[str, Any] | None = None,
    diagnostics_text: str | None = None,
    model: Any | None = None,
    tfidf: Any | None = None,
    svd: Any | None = None,
    feature_columns: list[str] | None = None,
) -> str | None:
    """Log one model from the comparison suite. Returns run_id or None.

    Params/metrics go to the Postgres tracking store; fitted ``model`` plus
    optional feature transformers are uploaded to the artifact store (MinIO).
    """
    tags = _base_tags(model_family=model_family, stage="compare")
    with start_run(
        experiment=EXPERIMENT_COMPARE,
        run_name=f"compare_{model_family}",
        tags=tags,
    ) as run:
        if run is None:
            return None
        import mlflow

        _log_params({
            **params,
            "n_train": n_train,
            "n_test": n_test,
        })
        _log_metrics(_improvement_metrics(metrics))

        if diagnostics is not None or diagnostics_text:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "ols_diagnostics.json"
                payload = {
                    "diagnostics": diagnostics,
                    "report": diagnostics_text,
                }
                path.write_text(json.dumps(payload, indent=2, default=str))
                mlflow.log_artifact(str(path))

        if model is not None:
            _log_model_artifact(model_family, model)

        if tfidf is not None and svd is not None and feature_columns is not None:
            _log_feature_artifacts(
                tfidf=tfidf, svd=svd, feature_columns=feature_columns
            )

        return run.info.run_id


def log_tune_trial(
    *,
    model_family: str,
    trial_number: int,
    params: dict[str, Any],
    valid_mae: float,
    pruned: bool = False,
) -> None:
    """Log a nested Optuna trial under the active parent tune run."""
    if not mlflow_enabled():
        return
    import mlflow

    # Must be called inside an active parent run
    with mlflow.start_run(
        run_name=f"trial_{trial_number:03d}",
        nested=True,
    ):
        mlflow.set_tags(_base_tags(model_family=model_family, stage="tune_trial"))
        _log_params({**params, "trial_number": trial_number, "pruned": pruned})
        mlflow.log_metric("valid_mae", float(valid_mae))


def log_tune_best(
    *,
    model_family: str,
    best_params: dict[str, Any],
    best_valid_mae: float,
    test_metrics: dict[str, float],
    config: dict[str, Any],
    n_train: int,
    n_valid: int,
    n_test: int,
    tuning_json_path: Path | None = None,
    model: Any | None = None,
    is_champion: bool = False,
) -> str | None:
    """Log best-params / test metrics onto the active parent tune run."""
    if not setup_tracking():
        return None

    import mlflow

    if mlflow.active_run() is None:
        _ensure_experiment(EXPERIMENT_TUNE)
        mlflow.start_run(run_name=f"tune_{model_family}_best")

    mlflow.set_tags(_base_tags(model_family=model_family, stage="tune_best"))
    if is_champion:
        mlflow.set_tag("champion", "true")
    _log_params({
        **{f"best_{k}": v for k, v in best_params.items()},
        **{f"cfg_{k}": v for k, v in config.items()},
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
    })
    metrics = _improvement_metrics(test_metrics)
    metrics["best_valid_mae"] = best_valid_mae
    _log_metrics(metrics)

    if tuning_json_path and Path(tuning_json_path).exists():
        mlflow.log_artifact(str(tuning_json_path))

    if model is not None:
        _log_model_artifact(model_family, model)

    run = mlflow.active_run()
    return run.info.run_id if run else None


def mark_champion(run_id: str) -> None:
    if not setup_tracking() or not run_id:
        return
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"].strip())
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "champion", "true")

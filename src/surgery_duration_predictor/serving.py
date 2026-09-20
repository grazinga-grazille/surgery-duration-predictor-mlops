"""Load serving artifacts from MLflow (MinIO) or fall back to local models/.

FastAPI uses this when ``MLFLOW_TRACKING_URI`` is set together with either:

- ``MLFLOW_MODEL_URI`` — e.g. ``runs:/<run_id>/model``
- ``MLFLOW_SERVING_MODEL_FAMILY`` — e.g. ``xgboost`` / ``random_forest``
  (resolves the latest finished compare run for that family that has ``features/``)

Each compare run is expected to store:

- ``model/`` — fitted estimator (xgboost or sklearn flavor)
- ``features/`` — ``tfidf.pkl``, ``svd.pkl``, ``feature_columns.pkl``
"""

from __future__ import annotations

import os
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from surgery_duration_predictor.artifacts import load_artifacts as load_local_artifacts
from surgery_duration_predictor.mlflow_logging import (
    EXPERIMENT_COMPARE,
    setup_tracking,
)


def _parse_run_id_from_model_uri(model_uri: str) -> str | None:
    """Extract run_id from ``runs:/<run_id>/model`` (optional trailing path)."""
    uri = model_uri.strip()
    if not uri.startswith("runs:/"):
        return None
    rest = uri[len("runs:/") :].strip("/")
    if not rest:
        return None
    return rest.split("/", 1)[0]


def _run_has_features(client: Any, run_id: str) -> bool:
    try:
        paths = {a.path for a in client.list_artifacts(run_id)}
    except Exception:
        return False
    return "features" in paths


def resolve_serving_run_id() -> tuple[str, str]:
    """Return ``(run_id, model_family)`` for the model FastAPI should serve."""
    if not setup_tracking():
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    explicit = os.getenv("MLFLOW_MODEL_URI", "").strip()
    if explicit:
        run_id = _parse_run_id_from_model_uri(explicit)
        if not run_id:
            raise ValueError(
                f"MLFLOW_MODEL_URI must look like runs:/<run_id>/model, got {explicit!r}"
            )
        if not _run_has_features(client, run_id):
            raise RuntimeError(
                f"Run {run_id} has no features/ artifacts. "
                "Re-run: uv run scripts/compare_models.py --skip-rf"
            )
        run = client.get_run(run_id)
        family = run.data.tags.get("model_family", "unknown")
        return run_id, family

    family = os.getenv("MLFLOW_SERVING_MODEL_FAMILY", "xgboost").strip()
    exp = client.get_experiment_by_name(EXPERIMENT_COMPARE)
    if exp is None:
        raise RuntimeError(f"Experiment {EXPERIMENT_COMPARE!r} not found")

    runs = client.search_runs(
        exp.experiment_id,
        filter_string=(
            f"tags.model_family = '{family}' AND attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    for run in runs:
        rid = run.info.run_id
        if _run_has_features(client, rid):
            return rid, family

    raise RuntimeError(
        f"No finished {family!r} compare run with features/ artifacts in "
        f"{EXPERIMENT_COMPARE!r}. Re-run: uv run scripts/compare_models.py --skip-rf"
    )


def _load_model(model_uri: str, model_family: str) -> Any:
    if model_family == "xgboost":
        import mlflow.xgboost

        return mlflow.xgboost.load_model(model_uri)
    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri)


def _load_feature_bundle(run_id: str) -> dict[str, Any]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(client.download_artifacts(run_id, "features", dst_path=tmp))
        out: dict[str, Any] = {}
        for name, key in (
            ("tfidf.pkl", "tfidf"),
            ("svd.pkl", "svd"),
            ("feature_columns.pkl", "feature_columns"),
        ):
            path = local / name
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {name} under features/ for run {run_id}. "
                    "Re-run compare_models.py so feature artifacts are logged."
                )
            with open(path, "rb") as f:
                out[key] = pickle.load(f)
        return out


@lru_cache(maxsize=1)
def load_serving_artifacts() -> dict[str, Any]:
    """Load model + feature transformers for online prediction.

    Prefers MLflow/MinIO when tracking is configured; otherwise local ``models/``.
    """
    use_mlflow = bool(os.getenv("MLFLOW_TRACKING_URI", "").strip()) and (
        bool(os.getenv("MLFLOW_MODEL_URI", "").strip())
        or bool(os.getenv("MLFLOW_SERVING_MODEL_FAMILY", "").strip())
        or os.getenv("MLFLOW_LOAD_FROM_ARTIFACTS", "").strip().lower()
        in {"1", "true", "yes"}
    )

    if not use_mlflow:
        arts = load_local_artifacts()
        return {
            "model": arts["rf_model"],
            "tfidf": arts["tfidf"],
            "svd": arts["svd"],
            "feature_columns": arts["feature_columns"],
            "model_family": "random_forest",
            "run_id": None,
            "source": "local",
        }

    run_id, family = resolve_serving_run_id()
    model_uri = os.getenv("MLFLOW_MODEL_URI", "").strip() or f"runs:/{run_id}/model"
    features = _load_feature_bundle(run_id)
    model = _load_model(model_uri, family)
    return {
        "model": model,
        "tfidf": features["tfidf"],
        "svd": features["svd"],
        "feature_columns": features["feature_columns"],
        "model_family": family,
        "run_id": run_id,
        "source": "mlflow",
    }

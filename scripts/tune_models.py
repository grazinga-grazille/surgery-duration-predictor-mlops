"""Optuna hyperparameter tuning for Random Forest and/or XGBoost.

Logs trials + best runs to MLflow experiment `surgery-duration-tune` when
MLFLOW_TRACKING_URI is set. When both families are tuned, tags the lower
test-MAE run as champion.

    uv run scripts/tune_models.py
    uv run scripts/tune_models.py --model xgboost
    uv run scripts/tune_models.py --model random_forest --n-trials 30
    uv run scripts/tune_models.py --no-mlflow
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from surgery_duration_predictor.data import load_data
from surgery_duration_predictor.mlflow_logging import mark_champion
from surgery_duration_predictor.tuning import (
    format_tuning_summary,
    load_tuning_config,
    save_tuning_result,
    tune_model,
)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "both"],
        default="both",
        help="Which model family to tune (default: both).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override config tuning.n_trials.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Optuna progress bar.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if MLFLOW_TRACKING_URI is set.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    if args.no_mlflow:
        os.environ.pop("MLFLOW_TRACKING_URI", None)

    cfg = load_tuning_config()
    n_trials = args.n_trials if args.n_trials is not None else int(cfg["n_trials"])

    print("Loading data ...")
    df = load_data()
    print(f"  {len(df):,} rows loaded.")
    print(
        f"  Tuning config: n_trials={n_trials}, "
        f"train_fraction={cfg['train_fraction']}, "
        f"valid_within_train={cfg['valid_within_train']}, "
        f"seed={cfg['random_seed']}"
    )
    if os.getenv("MLFLOW_TRACKING_URI"):
        print(f"  MLflow tracking: {os.environ['MLFLOW_TRACKING_URI']}")
    else:
        print("  MLflow tracking: disabled")
    print()

    kinds = (
        ["random_forest", "xgboost"]
        if args.model == "both"
        else [args.model]
    )

    results = []
    for kind in kinds:
        print(f"Starting Optuna study: {kind} ({n_trials} trials)")
        print("Objective: minimize validation MAE (minutes)")
        result = tune_model(
            df,
            kind,  # type: ignore[arg-type]
            n_trials=n_trials,
            tuning_cfg=cfg,
            show_progress=not args.no_progress,
            log_mlflow=not args.no_mlflow,
        )
        print()
        print(format_tuning_summary(result))
        # Avoid double-write when tune_model already saved under MLflow path
        out = result.get("tuning_json_path") or save_tuning_result(result)
        print(f"  Saved {out}")
        if result.get("mlflow_run_id"):
            print(f"  MLflow run: {result['mlflow_run_id']}")
        print()
        results.append(result)

    # Champion = lowest test MAE among tuned families in this invocation
    if len(results) >= 2:
        champion = min(results, key=lambda r: r["test_metrics"]["mae"])
        print(
            f"Champion (lowest test MAE): {champion['kind']} "
            f"(MAE={champion['test_metrics']['mae']:.2f} min)"
        )
        if champion.get("mlflow_run_id"):
            mark_champion(champion["mlflow_run_id"])
            print(f"  Tagged MLflow run {champion['mlflow_run_id']} as champion=true")


if __name__ == "__main__":
    main()

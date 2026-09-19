"""Optuna hyperparameter tuning for Random Forest and/or XGBoost.

Uses the same feature pipeline as production training. Hold-out protocol
matches the bike-demand notebook: 64% train / 16% valid / 20% test.

    uv run scripts/tune_models.py
    uv run scripts/tune_models.py --model xgboost
    uv run scripts/tune_models.py --model random_forest --n-trials 30
"""

from __future__ import annotations

import argparse
import warnings

from surgery_duration_predictor.data import load_data
from surgery_duration_predictor.tuning import (
    format_tuning_summary,
    load_tuning_config,
    save_tuning_result,
    tune_model,
)


def main() -> None:
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
    args = parser.parse_args()

    # Optuna/XGBoost can be chatty; keep the CLI readable.
    warnings.filterwarnings("ignore", category=UserWarning)

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
    print()

    kinds = (
        ["random_forest", "xgboost"]
        if args.model == "both"
        else [args.model]
    )

    for kind in kinds:
        print(f"Starting Optuna study: {kind} ({n_trials} trials)")
        print("Objective: minimize validation MAE (minutes)")
        result = tune_model(
            df,
            kind,  # type: ignore[arg-type]
            n_trials=n_trials,
            tuning_cfg=cfg,
            show_progress=not args.no_progress,
        )
        print()
        print(format_tuning_summary(result))
        out = save_tuning_result(result)
        print(f"  Saved {out}")
        print()


if __name__ == "__main__":
    main()

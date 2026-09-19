"""Compare Linear Regression (with OLS assumption checks) and XGBoost.

Uses the same feature pipeline and train/test split as production RF training.

    uv sync
    uv run scripts/compare_models.py
    uv run scripts/compare_models.py --skip-rf   # LR + XGBoost only (faster)
"""

from __future__ import annotations

import argparse

from surgery_duration_predictor.data import load_data
from surgery_duration_predictor.model_comparison import compare_models


def _print_summary(summary) -> None:
    print("── Test-set comparison (minutes scale) ─────────────────")
    print(
        f"{'model':<16} {'R²(min)':>8} {'R²(log)':>8} "
        f"{'RMSE':>8} {'MAE':>8} {'vs booked RMSE':>16}"
    )
    for _, row in summary.iterrows():
        rmse_imp = (1 - row["rmse"] / row["booked_rmse"]) * 100
        print(
            f"{row['model']:<16} {row['r2']:8.4f} {row['r2_log']:8.4f} "
            f"{row['rmse']:8.1f} {row['mae']:8.1f} {rmse_imp:15.1f}%"
        )
    booked_rmse = summary["booked_rmse"].iloc[0]
    booked_mae = summary["booked_mae"].iloc[0]
    print()
    print(f"  Booked baseline — RMSE={booked_rmse:.1f} min  MAE={booked_mae:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip Random Forest retrain (compare LR + XGBoost only).",
    )
    args = parser.parse_args()

    print("Loading data ...")
    df = load_data()
    print(f"  {len(df):,} rows loaded.\n")

    models = ("linear", "xgboost") if args.skip_rf else ("linear", "xgboost", "random_forest")
    result = compare_models(df, models=models, include_rf=not args.skip_rf)
    print()
    _print_summary(result["summary"])

    lr = result["models"].get("linear")
    if lr and lr.get("diagnostics"):
        ok = lr["diagnostics"]["ols_assumptions_ok"]
        print()
        if ok:
            print("Linear Regression assumptions held under the strict gate.")
        else:
            print(
                "Linear Regression assumptions did NOT hold. "
                "Prefer XGBoost / Random Forest for deployment."
            )


if __name__ == "__main__":
    main()

"""Build dashboard pickles: ML metrics + business analysis for Streamlit.

Aligns ML Analysis / Business Analysis with the serving model (XGBoost).

    uv run scripts/build_dashboard_artifacts.py
    uv run scripts/build_dashboard_artifacts.py --no-mlflow   # local XGB rebuild
"""

from __future__ import annotations

import argparse

from surgery_duration_predictor.dashboard_metrics import save_dashboard_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Ignore MLflow and rebuild Optuna-best XGBoost locally.",
    )
    args = parser.parse_args()

    print("Building dashboard artifacts ...")
    result = save_dashboard_artifacts(prefer_mlflow=not args.no_mlflow)
    meta = result["meta"]
    stats = result["model_stats"]

    print(f"  model: {stats.get('model_family')} ({stats.get('source')})")
    if stats.get("run_id"):
        print(f"  run_id: {stats['run_id']}")
    print(
        f"  holdout: R²={stats['r2']:.4f}  RMSE={stats['rmse']:.1f}  "
        f"MAE={stats['mae']:.1f}  n_test={stats['test_size']:,}"
    )
    print(
        f"  business ({meta['date_start']} → {meta['date_end']}): "
        f"{meta['n_cases']:,} cases"
    )
    print(
        f"  baseline ${meta['total_baseline_cost']:,.0f}  →  "
        f"model ${meta['total_predicted_cost']:,.0f}  "
        f"(savings ${meta['total_savings']:+,.0f})"
    )
    print(f"  additional procedures: {meta['total_additional']:,}")
    print(f"  wrote pickles → {result['output_dir']}")


if __name__ == "__main__":
    main()

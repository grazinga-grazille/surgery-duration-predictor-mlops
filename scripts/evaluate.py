"""Entry point: evaluate the saved model against the held-out test set.

    uv run scripts/evaluate.py

Loads the pre-saved model_stats and test_results artifacts produced by
scripts/train.py and prints a performance summary.
"""

from __future__ import annotations

import pickle
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    required = ["model_stats", "test_results"]
    for name in required:
        if not (MODELS_DIR / f"{name}.pkl").exists():
            raise SystemExit(
                f"'{name}.pkl' not found. Run `uv run scripts/train.py` first."
            )

    def _load(name: str):
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            return pickle.load(f)

    stats = _load("model_stats")
    results = _load("test_results")

    rmse_imp = (1 - stats["rmse"] / stats["booked_rmse"]) * 100
    mae_imp = (1 - stats["mae"] / stats["booked_mae"]) * 100
    under = (results["residual"] < 0).mean() * 100
    over = (results["residual"] > 0).mean() * 100

    print("── Model Performance ───────────────────────────────")
    print(f"  R²:          {stats['r2']:.4f}")
    print(f"  RMSE:        {stats['rmse']:.1f} min")
    print(f"  MAE:         {stats['mae']:.1f} min")
    print(f"  Test cases:  {stats['test_size']:,}")
    print()
    print("── vs. Booked Duration Baseline ────────────────────")
    print(f"  Booked RMSE: {stats['booked_rmse']:.1f} min  (model improves by {rmse_imp:.1f}%)")
    print(f"  Booked MAE:  {stats['booked_mae']:.1f} min  (model improves by {mae_imp:.1f}%)")
    print()
    print(f"  Under-predicted: {under:.1f}%  |  Over-predicted: {over:.1f}%")


if __name__ == "__main__":
    main()

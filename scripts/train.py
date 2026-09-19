"""Entry point: train the model and save all artifacts to models/.

    uv run scripts/train.py

All real logic lives in src/surgery_duration_predictor/. This script just
wires things together, saves each artifact as a .pkl, and prints a summary.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from surgery_duration_predictor.data import load_data
from surgery_duration_predictor.train import train_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    print("Loading data ...")
    df = load_data()
    print(f"  {len(df):,} rows loaded.")

    print("Training model ...")
    artifacts = train_model(df)

    MODELS_DIR.mkdir(exist_ok=True)
    for name, obj in artifacts.items():
        out_path = MODELS_DIR / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {out_path.name}")

    stats = artifacts["model_stats"]
    print(
        f"\nDone — R²={stats['r2']:.3f}  "
        f"RMSE={stats['rmse']:.1f} min  "
        f"MAE={stats['mae']:.1f} min"
    )


if __name__ == "__main__":
    main()

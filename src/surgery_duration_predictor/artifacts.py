"""Artifact loading.

Loads all .pkl model artifacts once at startup and caches them in memory
via lru_cache. Both the FastAPI app and the Streamlit app use this module.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    """Load and cache all model artifacts from models/.

    Returns
    -------
    dict with keys: rf_model, tfidf, svd, feature_columns, categorical_values,
                    valid_combinations, procedure_examples, model_stats, test_results
    """
    def _load(name: str):
        with open(MODELS_DIR / name, "rb") as f:
            return pickle.load(f)

    return {
        "rf_model":          _load("rf_model.pkl"),
        "tfidf":             _load("tfidf.pkl"),
        "svd":               _load("svd.pkl"),
        "feature_columns":   _load("feature_columns.pkl"),
        "categorical_values": _load("categorical_values.pkl"),
        "valid_combinations": _load("valid_combinations.pkl"),
        "procedure_examples": _load("procedure_examples.pkl"),
        "model_stats":       _load("model_stats.pkl"),
        "test_results":      _load("test_results.pkl"),
    }

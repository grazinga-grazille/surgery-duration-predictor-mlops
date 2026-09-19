"""
Step 7 — Build final TF-IDF embeddings for duration prediction.

Uses scikit-learn's standard ``TfidfVectorizer`` (log-damped IDF is fine here).
The Sarica & Luo harsher TF-IDF variant is only for Step 3 stopword discovery.

By default, input is the finalized preprocess output
(``data/interim/cases_cleaned.parquet``): phrase-joined, baseline-stopword-
filtered ``cleaned_text``. Tune ``TFIDF_*`` knobs in ``config.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .utils import ensure_dirs, save_text_list, setup_logging

logger = logging.getLogger("surgical_stopwords.embeddings")


def build_tfidf(
    texts: List[str],
    max_features: Optional[int] = config.TFIDF_MAX_FEATURES,
    ngram_range=config.TFIDF_NGRAM_RANGE,
    min_df: Union[int, float] = config.TFIDF_MIN_DF,
    max_df: Union[int, float] = config.TFIDF_MAX_DF,
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Fit and transform a sparse TF-IDF matrix on cleaned procedure text (Step 7).

    ``token_pattern`` keeps underscore-joined phrases, hyphens, and glued
    ampersands (``t&a``, ``a&p``) as single features.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,  # already lowercased in Step 1
        # Keep underscore phrases, hyphens, and glued ampersands (t&a, a&p)
        token_pattern=r"(?u)\b\w[\w\-/&]*\w\b|\b\w\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def run(
    input_path: Optional[Union[str, Path]] = None,
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Execute Step 7: fit/transform TF-IDF and persist artifacts under data/processed/.

    Artifacts
    ---------
    - ``data/processed/tfidf_matrix.npz`` — sparse CSR matrix (n_cases x n_features)
    - ``data/processed/tfidf_vectorizer.joblib`` — fitted vectorizer for new cases
    - ``data/processed/tfidf_feature_names.txt`` — vocabulary, one term per line
    - ``data/processed/tfidf_row_index.parquet`` — case_id (+ duration) per matrix row
    """
    setup_logging()
    ensure_dirs([config.DATA_PROCESSED])

    path = Path(input_path) if input_path else Path(config.TFIDF_INPUT_CASES_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"TF-IDF input not found: {path}. Run preprocess + phrases first."
        )

    cases = pd.read_parquet(path)
    required = {config.CASE_ID_COLUMN, "cleaned_text"}
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")

    texts = cases["cleaned_text"].fillna("").astype(str).tolist()
    logger.info("TF-IDF input: %s (%d documents)", path, len(texts))
    logger.info(
        "Params: max_features=%s min_df=%s max_df=%s ngram_range=%s",
        config.TFIDF_MAX_FEATURES,
        config.TFIDF_MIN_DF,
        config.TFIDF_MAX_DF,
        config.TFIDF_NGRAM_RANGE,
    )

    matrix, vectorizer = build_tfidf(texts)
    feature_names = list(vectorizer.get_feature_names_out())

    sparse.save_npz(config.TFIDF_MATRIX_PATH, matrix)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
    save_text_list(feature_names, config.TFIDF_FEATURE_NAMES_PATH)

    meta_cols = [config.CASE_ID_COLUMN]
    if config.DURATION_COLUMN in cases.columns:
        meta_cols.append(config.DURATION_COLUMN)
    if "EncounterID" in cases.columns:
        meta_cols.append("EncounterID")
    cases[meta_cols].to_parquet(config.TFIDF_ROW_INDEX_PATH, index=False)

    logger.info(
        "TF-IDF matrix shape=%s nnz=%d density=%.6f",
        matrix.shape,
        matrix.nnz,
        matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1),
    )
    logger.info("Wrote matrix      -> %s", config.TFIDF_MATRIX_PATH)
    logger.info("Wrote vectorizer  -> %s", config.TFIDF_VECTORIZER_PATH)
    logger.info(
        "Wrote features    -> %s (%d names)",
        config.TFIDF_FEATURE_NAMES_PATH,
        len(feature_names),
    )
    logger.info("Wrote row index   -> %s", config.TFIDF_ROW_INDEX_PATH)
    return matrix, vectorizer


if __name__ == "__main__":
    run()

"""Feature engineering.

Converts raw surgical case DataFrames into model-ready feature matrices.
All feature logic lives here so scripts, notebooks, and tests share one
source of truth.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Constants ─────────────────────────────────────────────────────────────────
TEXT_COL = "ProcedureDescription"
TARGET_COL = "ActualDurationMinutes"
BOOKED_COL = "book_dur"
SPECIALTY_COL = "ProcedureSpecialtyDescription"

SVD_N_COMPONENTS = 150
TFIDF_MAX_FEATURES = 5000

# Baseline stopwords from the surgical_stopwords pipeline (EXTRA_BASELINE_STOPWORDS
# in config.py). Applied after slash normalisation so e.g. "w/o" → "" → "wo"
# never reaches the token filter as a stray "w" or "o".
_BASELINE_STOP_WORDS: frozenset[str] = frozenset({
    "with", "wo", "and", "of", "s",
})

_ALL_STOP_WORDS: frozenset[str] = _BASELINE_STOP_WORDS

# Compiled once at import time.
_SIDE_MARKER_RE = re.compile(r"\(\s*s\s*\)", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(txt: str) -> str:
    """Normalise and clean a procedure description, matching the offline pipeline.

    Order mirrors preprocess.py (normalize_text) then apply_stopwords.py:
      1. Lowercase and remove the side-marker "(S)"
      2. Expand slash abbreviations (w/o, w/, /) before stripping non-alpha —
         this ensures "w/o" is removed entirely rather than becoming "w o"
      3. Strip remaining non-alpha characters
      4. Remove baseline stops (with, wo, and, of, s) then domain stop-words
    """
    txt = str(txt).lower().strip()

    # 1. Side marker — remove "(S)" / "(s)" as a unit before any other stripping
    txt = _SIDE_MARKER_RE.sub(" ", txt)

    # 2. Slash normalisation (longer patterns first to avoid partial matches)
    txt = txt.replace("w/o", " ")
    txt = txt.replace("w/", " ")
    txt = txt.replace("/", " ")

    # 3. Strip non-alpha, non-space characters
    txt = re.sub(r"[^a-z ]", " ", txt)
    txt = _MULTI_SPACE_RE.sub(" ", txt).strip()

    # 4. Remove baseline + domain stop-words
    return " ".join(w for w in txt.split() if w not in _ALL_STOP_WORDS)


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer | None = None,
    svd: TruncatedSVD | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, TfidfVectorizer, TruncatedSVD]:
    """Turn a raw DataFrame into a feature matrix X and a log-transformed target y.

    Parameters
    ----------
    df   : Raw DataFrame from load_data().
    tfidf: Pre-fitted TfidfVectorizer. Pass None only when fit=True.
    svd  : Pre-fitted TruncatedSVD. Pass None only when fit=True.
    fit  : If True, fit tfidf and svd on this data (training set only).

    Returns
    -------
    X (pd.DataFrame), y (pd.Series, log scale), tfidf, svd
    """
    texts = df[TEXT_COL].apply(clean_text)

    if fit:
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2)
        )
        tfidf_mat = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
    else:
        tfidf_mat = tfidf.transform(texts)
        svd_mat = svd.transform(tfidf_mat)

    svd_df = pd.DataFrame(
        svd_mat,
        columns=[f"SVD_{i}" for i in range(SVD_N_COMPONENTS)],
        index=df.index,
    )

    X_num = pd.DataFrame(
        {"SurgicalPriority": df["SurgicalPriority"].values}, index=df.index
    )
    X_cat_raw = df[
        ["PatientType", "Roomdescription", "ProcedureSpecialtyDescription"]
    ].copy()
    X_cat = pd.get_dummies(X_cat_raw, drop_first=True, dtype=int)

    X = pd.concat([X_num, X_cat, svd_df], axis=1)
    y = np.log(df[TARGET_COL])

    return X, y, tfidf, svd

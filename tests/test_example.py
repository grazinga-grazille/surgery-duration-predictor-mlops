"""Package smoke tests.

Because logic lives in importable functions (not notebook cells), it's
testable without opening a notebook or loading data.
"""

import re

import numpy as np
import pandas as pd
import pytest

import surgery_duration_predictor
from surgery_duration_predictor.features import clean_text, build_features, SVD_N_COMPONENTS


def test_package_imports():
    assert surgery_duration_predictor.__version__


def test_clean_text_lowercases():
    assert clean_text("ARTHROSCOPY") == "arthroscopy"


def test_clean_text_removes_domain_stop_words():
    result = clean_text("bilateral removal of total knee")
    assert "bilateral" not in result
    assert "removal" not in result
    assert "total" not in result
    assert "knee" in result


def test_clean_text_removes_baseline_stop_words():
    # "of", "with", "and" are baseline stops — must be removed
    result = clean_text("repair of the knee with fixation and plate")
    assert "of" not in result
    assert "with" not in result
    assert "and" not in result
    assert "knee" in result


def test_clean_text_slash_before_strip():
    # "w/o" must be removed entirely, not fragmented into "w" and "o"
    result = clean_text("w/o removal of bilateral knee")
    assert "w" not in result.split()
    assert "o" not in result.split()
    assert "knee" in result

    # "w/" similarly must disappear before non-alpha stripping
    result2 = clean_text("repair w/ plate fixation")
    assert "w" not in result2.split()
    assert "plate" in result2


def test_clean_text_side_marker_removed():
    # "(S)" should be dropped as a unit, not leave a stray "s" token
    result = clean_text("knee arthroscopy (S)")
    assert "s" not in result.split()
    assert "knee" in result


def test_clean_text_strips_punctuation():
    assert re.fullmatch(r"[a-z ]*", clean_text("knee (left) repair, #2"))


def test_build_features_fit():
    df = pd.DataFrame({
        "ProcedureDescription": ["knee arthroscopy", "hip replacement", "appendectomy"],
        "SurgicalPriority": [2, 3, 1],
        "PatientType": ["Inpatient", "Outpatient", "Inpatient"],
        "Roomdescription": ["OR1", "OR2", "OR1"],
        "ProcedureSpecialtyDescription": ["Ortho", "Ortho", "General"],
        "ActualDurationMinutes": [90.0, 120.0, 45.0],
        "book_dur": [80.0, 100.0, 50.0],
    })
    X, y, tfidf, svd = build_features(df, fit=True)
    assert len(X) == 3
    assert len(y) == 3
    assert X.shape[1] > SVD_N_COMPONENTS  # numeric + categorical + SVD cols
    assert np.allclose(y, np.log(df["ActualDurationMinutes"]))

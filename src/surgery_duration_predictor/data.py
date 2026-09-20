"""Data loading and cleaning.

Reads the pre-processed surgical case CSV (LoadFile.csv) from data/01_raw/
and returns a clean DataFrame ready for feature engineering.

Cases with ScheduledDate after ML_CUTOFF_DATE are excluded here so that
train/test/tune all share one ML window. Post-cutoff rows are reserved for
future monitoring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "01_raw"

# Inclusive end of the ML modeling window (train + test). Later dates → monitoring.
ML_CUTOFF_DATE = pd.Timestamp("2024-04-27")

_KEY_COLS = [
    "SurgicalPriority",
    "PatientType",
    "Roomdescription",
    "ProcedureSpecialtyDescription",
    "ProcedureDescription",
    "ActualDurationMinutes",
    "book_dur",
]


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load and return the surgical case dataset for ML (through ML_CUTOFF_DATE).

    Parameters
    ----------
    path : Path to LoadFile.csv. Defaults to data/01_raw/LoadFile.csv.

    Returns
    -------
    pd.DataFrame with one row per surgical case, ready for feature engineering.
    """
    if path is None:
        path = RAW_DIR / "LoadFile.csv"

    df = pd.read_csv(path, low_memory=False)

    # Keep only rows with valid actual duration and booked duration
    df = df[df["ActualDurationMinutes"] > 0]
    df = df[df["book_dur"] > 0]

    # Drop rows missing any key feature
    df = df.dropna(subset=_KEY_COLS)

    # ML window only — post-cutoff cases reserved for monitoring
    scheduled = pd.to_datetime(df["ScheduledDate"], errors="coerce")
    df = df.loc[scheduled.notna() & (scheduled <= ML_CUTOFF_DATE)]

    return df.reset_index(drop=True)


def load_analysis_data(
    path: Path | None = None,
    *,
    date_start: str = "2024-03-01",
    date_end: str = "2025-03-31",
) -> pd.DataFrame:
    """Load cases for business analysis (date window, no ML cutoff).

    Used for financial / resource utilization charts on the dashboard.
    Applies the same duration / key-column filters as ``load_data``.
    """
    if path is None:
        path = RAW_DIR / "LoadFile.csv"

    df = pd.read_csv(path, low_memory=False)
    df = df[df["ActualDurationMinutes"] > 0]
    df = df[df["book_dur"] > 0]
    df = df.dropna(subset=_KEY_COLS)

    scheduled = pd.to_datetime(df["ScheduledDate"], errors="coerce")
    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    df = df.loc[scheduled.notna() & (scheduled >= start) & (scheduled <= end)]
    return df.reset_index(drop=True)

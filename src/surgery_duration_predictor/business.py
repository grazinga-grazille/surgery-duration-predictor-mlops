"""Business impact calculations for the Streamlit dashboard.

Cost formula (per case, always non-negative), from config/cost:

  net = actual - planned
  if net >= 0:  # over-ran (overtime)
      cost = net * undertime_rate * overtime_multiplier
  else:         # under-ran (opportunity / idle)
      cost = abs(net) * undertime_rate

Financial impact compares booked vs model-predicted planned durations.
Resource utilization converts recovered under-run minutes into additional
high-volume procedures that fit in that idle OR time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from surgery_duration_predictor.features import (
    BOOKED_COL,
    SPECIALTY_COL,
    TARGET_COL,
    TEXT_COL,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

DEFAULT_ANALYSIS = {
    "date_start": "2024-03-01",
    "date_end": "2025-03-31",
}
DEFAULT_COST = {
    "undertime_rate": 35.0,
    "overtime_multiplier": 1.5,
}

# Procedures with at least this many cases in a specialty count as "high-volume"
# for the recovered-time capacity estimate.
HIGH_VOLUME_MIN_COUNT = 10


def load_analysis_config(path: Path | None = None) -> dict[str, Any]:
    """Load analysis date window + OR cost rates from config.yaml."""
    analysis = dict(DEFAULT_ANALYSIS)
    cost = dict(DEFAULT_COST)
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        analysis.update(raw.get("analysis") or {})
        cost.update(raw.get("cost") or {})
    return {"analysis": analysis, "cost": cost}


def case_scheduling_cost(
    actual: np.ndarray | pd.Series,
    planned: np.ndarray | pd.Series,
    *,
    undertime_rate: float = 35.0,
    overtime_multiplier: float = 1.5,
) -> np.ndarray:
    """Per-case OR scheduling cost from actual vs planned duration (minutes)."""
    actual_a = np.asarray(actual, dtype=float)
    planned_a = np.asarray(planned, dtype=float)
    net = actual_a - planned_a
    overtime = net >= 0
    cost = np.empty_like(net, dtype=float)
    cost[overtime] = net[overtime] * undertime_rate * overtime_multiplier
    cost[~overtime] = np.abs(net[~overtime]) * undertime_rate
    return cost


def financial_impact_by_specialty(
    df: pd.DataFrame,
    predicted: np.ndarray | pd.Series,
    *,
    undertime_rate: float = 35.0,
    overtime_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Aggregate baseline (booked) vs model cost by specialty."""
    actual = df[TARGET_COL].to_numpy(dtype=float)
    booked = df[BOOKED_COL].to_numpy(dtype=float)
    pred = np.asarray(predicted, dtype=float)

    baseline = case_scheduling_cost(
        actual, booked,
        undertime_rate=undertime_rate,
        overtime_multiplier=overtime_multiplier,
    )
    model_cost = case_scheduling_cost(
        actual, pred,
        undertime_rate=undertime_rate,
        overtime_multiplier=overtime_multiplier,
    )

    out = (
        pd.DataFrame({
            SPECIALTY_COL: df[SPECIALTY_COL].values,
            "baseline_cost": baseline,
            "predicted_cost": model_cost,
            "case_count": 1,
        })
        .groupby(SPECIALTY_COL, as_index=False)
        .agg(
            baseline_cost=("baseline_cost", "sum"),
            predicted_cost=("predicted_cost", "sum"),
            case_count=("case_count", "sum"),
        )
    )
    out["savings"] = out["baseline_cost"] - out["predicted_cost"]
    return out.sort_values("baseline_cost", ascending=False).reset_index(drop=True)


def resource_utilization_by_specialty(
    df: pd.DataFrame,
    *,
    high_volume_min_count: int = HIGH_VOLUME_MIN_COUNT,
) -> pd.DataFrame:
    """Estimate extra high-volume cases that fit in recovered under-run OR time.

    For each specialty × OR room × day:
      net_recovered = sum(booked) - sum(actual)   (only if positive)
    Reference duration = mean actual duration of procedures with at least
    ``high_volume_min_count`` cases in that specialty.
    Additional possible = floor(net_recovered / reference), summed over room-days.
    """
    need = [SPECIALTY_COL, TEXT_COL, TARGET_COL, BOOKED_COL, "ScheduledDate", "Roomdescription"]
    work = df[need].copy()
    work["ScheduledDate"] = pd.to_datetime(work["ScheduledDate"], errors="coerce")
    work = work.dropna(subset=["ScheduledDate", "Roomdescription"])
    work["day"] = work["ScheduledDate"].dt.normalize()

    rows: list[dict[str, Any]] = []
    for specialty, g in work.groupby(SPECIALTY_COL):
        total_surgeries = int(len(g))
        proc_counts = g[TEXT_COL].value_counts()
        high_vol = proc_counts[proc_counts >= high_volume_min_count].index
        if len(high_vol) == 0:
            avg_dur = float(g[TARGET_COL].mean())
        else:
            avg_dur = float(g.loc[g[TEXT_COL].isin(high_vol), TARGET_COL].mean())
        if not np.isfinite(avg_dur) or avg_dur <= 0:
            avg_dur = float(g[TARGET_COL].median()) or 1.0

        additional = 0
        recovered_total = 0.0
        for _, day_g in g.groupby(["Roomdescription", "day"]):
            net = float(day_g[BOOKED_COL].sum() - day_g[TARGET_COL].sum())
            if net <= 0:
                continue
            recovered_total += net
            additional += int(net // avg_dur)

        rows.append({
            SPECIALTY_COL: specialty,
            "total_surgeries": total_surgeries,
            "avg_procedure_duration_min": avg_dur,
            "total_additional_possible": additional,
            "recovered_minutes": recovered_total,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("total_surgeries", ascending=False).reset_index(drop=True)

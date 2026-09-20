"""Unit tests for OR cost and resource utilization helpers."""

import pandas as pd

from surgery_duration_predictor.business import (
    case_scheduling_cost,
    financial_impact_by_specialty,
    resource_utilization_by_specialty,
)


def test_case_cost_undertime_and_overtime():
    # actual < planned → undertime at $35/min
    # actual > planned → overtime at $35 * 1.5
    actual = [50.0, 100.0]
    planned = [80.0, 70.0]
    cost = case_scheduling_cost(actual, planned)
    assert cost[0] == 30.0 * 35.0
    assert cost[1] == 30.0 * 35.0 * 1.5


def test_financial_impact_aggregates_by_specialty():
    df = pd.DataFrame({
        "ProcedureSpecialtyDescription": ["Ortho", "Ortho", "General"],
        "ActualDurationMinutes": [60.0, 90.0, 50.0],
        "book_dur": [80.0, 70.0, 50.0],
    })
    predicted = [70.0, 85.0, 55.0]
    out = financial_impact_by_specialty(df, predicted)
    assert set(out["ProcedureSpecialtyDescription"]) == {"Ortho", "General"}
    assert "savings" in out.columns
    assert out["case_count"].sum() == 3


def test_resource_utilization_room_day_net():
    df = pd.DataFrame({
        "ProcedureSpecialtyDescription": ["Ortho"] * 3,
        "ProcedureDescription": ["knee"] * 3,
        "ActualDurationMinutes": [40.0, 40.0, 40.0],
        "book_dur": [80.0, 80.0, 40.0],
        "ScheduledDate": ["2024-06-01", "2024-06-01", "2024-06-02"],
        "Roomdescription": ["OR1", "OR1", "OR1"],
    })
    # avg=40; Day1 net=+80 → floor(80/40)=2; Day2 net=0 → 0
    out = resource_utilization_by_specialty(df, high_volume_min_count=1)
    assert int(out.loc[0, "total_additional_possible"]) == 2

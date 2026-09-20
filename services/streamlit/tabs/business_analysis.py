"""tabs/business_analysis.py — Business Analysis tab

Uses precomputed summaries from ``scripts/build_dashboard_artifacts.py``
(models/dashboard_*.pkl). Cost formula (per case, always positive):

  net >= 0  (over-ran)  ->  net      x $35 x 1.5
  net <  0  (under-ran) ->  abs(net) x $35
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
_BLUE = "#4A90D9"
_TEAL = "#1ABC9C"


def render(financial: pd.DataFrame | None = None, resource: pd.DataFrame | None = None,
           meta: dict | None = None):
    st.subheader("Business Analysis")
    st.caption(
        "Quantifying the operational and financial impact of improved scheduling accuracy."
    )

    if financial is None or resource is None or financial.empty or resource.empty:
        st.info(
            "Business analysis artifacts not found. Run "
            "`uv run scripts/build_dashboard_artifacts.py` to compute financial "
            "and resource summaries from the serving model."
        )
        return

    meta = meta or {}
    date_start = meta.get("date_start", "2024-03-01")
    date_end = meta.get("date_end", "2025-03-31")

    fi_tab, ru_tab = st.tabs([
        "💰 Financial Impact",
        "🏥 Resource Utilization Impact",
    ])

    with fi_tab:
        _render_financial_impact(financial, date_start, date_end, meta)

    with ru_tab:
        _render_resource_utilization(resource, date_start, date_end)


# ── Chart helper ──────────────────────────────────────────────────────────────

def _bar_chart(specialties, values, color, y_title, key):
    fig = go.Figure(go.Bar(
        x=specialties,
        y=values,
        marker_color=color,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Cost: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title=y_title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", tickformat="$,.0f",
                   rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Financial impact renderer ─────────────────────────────────────────────────

def _render_financial_impact(data: pd.DataFrame, date_start: str, date_end: str, meta: dict):
    specialties = data["ProcedureSpecialtyDescription"].tolist()
    baseline_cost = data["baseline_cost"].tolist()
    pred_cost = data["predicted_cost"].tolist()
    savings = data["savings"].tolist()
    total_cases = int(data["case_count"].sum())
    total_saved = sum(savings)
    rate = float(meta.get("undertime_rate", 35.0))
    over_mult = float(meta.get("overtime_multiplier", 1.5))
    overtime_rate = rate * over_mult

    caption = f"**{date_start}** to **{date_end}** · {total_cases:,} cases"

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total Baseline Cost",
        f"${sum(baseline_cost):,.0f}",
        help=(
            "Total net OR scheduling cost using booked duration as the plan. "
            f"Overtime at \\${overtime_rate:.2f}/min, undertime at \\${rate:.0f}/min."
        ),
    )
    m2.metric(
        "Total Predicted Cost",
        f"${sum(pred_cost):,.0f}",
        help=(
            "Total net OR scheduling cost if the model's predicted duration had been "
            "used as the planned schedule instead of the booked duration."
        ),
    )
    m3.metric(
        "Net Cost Impact",
        f"${total_saved:+,.0f}",
        delta="savings from model predictions" if total_saved >= 0 else "additional cost from model predictions",
        delta_color="normal" if total_saved >= 0 else "inverse",
        help=(
            "Difference between baseline cost and predicted cost (baseline - predicted). "
            "Positive means the model reduces overall OR scheduling cost."
        ),
    )
    st.divider()

    st.markdown("#### Baseline Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the originally booked duration as the plan"
    )
    _bar_chart(specialties, baseline_cost, _BLUE, "Net OR Cost ($)", "fi_baseline")

    st.markdown("#### Model Prediction Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the model's predicted duration as the plan"
    )
    _bar_chart(specialties, pred_cost, _TEAL, "Net OR Cost ($)", "fi_predicted")

    st.markdown("#### Cost Savings by Specialty")
    st.caption(
        "How much the model reduces (or increases) net OR scheduling cost per specialty "
        "compared to the baseline. Green = model saves money · Red = model adds cost"
    )
    savings_colors = ["#27AE60" if s >= 0 else "#E74C3C" for s in savings]
    fig_sav = go.Figure(go.Bar(
        x=specialties,
        y=savings,
        marker_color=savings_colors,
        text=[f"${s:+,.0f}" for s in savings],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Cost Savings: $%{y:+,.0f}<extra></extra>",
    ))
    fig_sav.add_hline(y=0, line_color="#95A5A6", line_width=1)
    fig_sav.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Cost Savings vs. Baseline ($)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", tickformat="$,.0f"),
    )
    st.plotly_chart(fig_sav, use_container_width=True)


# ── Resource Utilization renderer ─────────────────────────────────────────────

def _render_resource_utilization(summary: pd.DataFrame, date_start: str, date_end: str):
    specialties = summary["ProcedureSpecialtyDescription"].tolist()
    performed = summary["total_surgeries"].tolist()
    additional = summary["total_additional_possible"].tolist()
    total_cases = sum(performed)
    total_extra = sum(additional)

    st.caption(
        f"**{date_start}** to **{date_end}** · {total_cases:,} surgeries performed · "
        f"{total_extra:,} additional surgeries possible from recovered schedule time"
    )

    m1, m2 = st.columns(2)
    m1.metric("Total Surgeries Performed", f"{total_cases:,}")
    m2.metric(
        "Additional Surgeries Possible",
        f"{total_extra:,}",
        help="High-count procedures that fit within recovered OR time across all days.",
    )
    st.divider()

    st.markdown("#### Total Surgeries Performed by Specialty")
    fig1 = go.Figure(go.Bar(
        x=specialties,
        y=performed,
        marker_color=_BLUE,
        text=performed,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Surgeries Performed: %{y:,}<extra></extra>",
    ))
    fig1.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Number of Surgeries",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", rangemode="tozero"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Additional Surgeries Possible from Recovered Schedule Time")
    st.caption(
        "Number of additional high-volume procedures that fit within net OR time recovered "
        "on room-days that finished under their booked schedule "
        "(floor of net under-run minutes ÷ mean high-volume procedure duration)."
    )
    fig2 = go.Figure(go.Bar(
        x=specialties,
        y=additional,
        marker_color=_TEAL,
        text=additional,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Additional Possible: %{y:,}<extra></extra>",
    ))
    fig2.update_layout(
        height=420,
        margin=dict(t=60, b=20, l=10, r=10),
        xaxis_title="Surgical Specialty",
        yaxis_title="Additional Surgeries",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f4f8"),
        xaxis=dict(tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(gridcolor="rgba(149,165,166,0.2)", rangemode="tozero"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Specialty Breakdown")
    ref = summary[[
        "ProcedureSpecialtyDescription",
        "avg_procedure_duration_min",
        "total_additional_possible",
    ]].copy()
    ref.columns = ["Specialty", "Avg Procedure Duration (min)", "Additional Surgeries Possible"]
    ref["Avg Procedure Duration (min)"] = ref["Avg Procedure Duration (min)"].round(1)
    st.dataframe(ref, use_container_width=True, hide_index=True)

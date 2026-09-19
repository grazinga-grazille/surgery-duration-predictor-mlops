"""tabs/business_analysis.py — Business Analysis tab

Reads precomputed aggregated summaries from Streamlit Secrets (no patient-level data).
Cost formula used during precomputation (applied per case, always positive):
  net >= 0  (over-ran)  ->  net      x $35 x 1.5
  net <  0  (under-ran) ->  abs(net) x $35
"""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
DATE_START = "2024-03-01"
DATE_END   = "2025-03-31"
_RATE      = 35.0
_RATE_OVER = _RATE * 1.5

_BLUE = "#4A90D9"
_TEAL = "#1ABC9C"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading analysis data ...")
def _load_comparison_data() -> pd.DataFrame:
    """Read precomputed financial impact summary from Streamlit Secrets."""
    return pd.read_csv(io.StringIO(st.secrets["financial_impact_csv"]))


@st.cache_data(show_spinner="Loading resource utilization data ...")
def _load_resource_utilization_data() -> pd.DataFrame:
    """Read precomputed resource utilization summary from Streamlit Secrets."""
    return pd.read_csv(io.StringIO(st.secrets["resource_utilization_csv"]))


# ── Tab renderer ──────────────────────────────────────────────────────────────

def render():
    st.subheader("Business Analysis")
    st.caption(
        "Quantifying the operational and financial impact of improved scheduling accuracy."
    )

    try:
        _ = st.secrets["financial_impact_csv"]
        _ = st.secrets["resource_utilization_csv"]
    except Exception:
        st.info(
            "Business analysis data is not configured for this environment. "
            "Add `financial_impact_csv` and `resource_utilization_csv` to "
            "`.streamlit/secrets.toml` to enable this tab. "
            "Prediction and ML Analysis still work without it."
        )
        return

    fi_tab, ru_tab = st.tabs([
        "💰 Financial Impact",
        "🏥 Resource Utilization Impact",
    ])

    with fi_tab:
        _render_financial_impact()

    with ru_tab:
        _render_resource_utilization()


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

def _render_financial_impact():
    data = _load_comparison_data()
    specialties   = data["ProcedureSpecialtyDescription"].tolist()
    baseline_cost = data["baseline_cost"].tolist()
    pred_cost     = data["predicted_cost"].tolist()
    savings       = data["savings"].tolist()
    total_cases   = int(data["case_count"].sum())
    total_saved   = sum(savings)

    caption = f"**{DATE_START}** to **{DATE_END}** · {total_cases:,} cases"

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Total Baseline Cost",
        f"${sum(baseline_cost):,.0f}",
        help=(
            "Total net OR scheduling cost across all specialties using the originally "
            "booked duration as the planned schedule. Each case is costed by how far "
            "its actual duration deviated from the booked time — overtime at \\$52.50 per min, "
            "undertime (opportunity cost) at \\$35 per min."
        ),
    )
    m2.metric(
        "Total Predicted Cost",
        f"${sum(pred_cost):,.0f}",
        help=(
            "Total net OR scheduling cost if the model's predicted duration had been "
            "used as the planned schedule instead of the booked duration. A lower value "
            "than the baseline means the model's predictions lead to smaller deviations "
            "from actual surgery times."
        ),
    )
    m3.metric(
        "Net Cost Impact",
        f"${total_saved:+,.0f}",
        delta="savings from model predictions" if total_saved >= 0 else "additional cost from model predictions",
        delta_color="normal" if total_saved >= 0 else "inverse",
        help=(
            "Difference between baseline cost and predicted cost (baseline - predicted). "
            "A positive value means the model's schedule predictions reduce overall OR "
            "scheduling cost compared to the original booked durations."
        ),
    )
    st.divider()

    # ── Baseline chart ────────────────────────────────────────────────────────
    st.markdown("#### Baseline Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the originally booked duration as the plan"
    )
    _bar_chart(specialties, baseline_cost, _BLUE, "Net OR Cost ($)", "fi_baseline")

    # ── Prediction chart ──────────────────────────────────────────────────────
    st.markdown("#### Model Prediction Net OR Cost by Specialty")
    st.caption(
        caption + " · Cost of scheduling deviations using the model's predicted duration as the plan"
    )
    _bar_chart(specialties, pred_cost, _TEAL, "Net OR Cost ($)", "fi_predicted")

    # ── Cost impact chart ─────────────────────────────────────────────────────
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

def _render_resource_utilization():
    summary = _load_resource_utilization_data()

    specialties  = summary["ProcedureSpecialtyDescription"].tolist()
    performed    = summary["total_surgeries"].tolist()
    additional   = summary["total_additional_possible"].tolist()
    total_cases  = sum(performed)
    total_extra  = sum(additional)

    st.caption(
        f"**{DATE_START}** to **{DATE_END}** · {total_cases:,} surgeries performed · "
        f"{total_extra:,} additional surgeries possible from recovered schedule time"
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2 = st.columns(2)
    m1.metric("Total Surgeries Performed", f"{total_cases:,}")
    m2.metric("Additional Surgeries Possible", f"{total_extra:,}",
              help="High-count procedures that fit within recovered OR time across all days.")
    st.divider()

    # ── Total surgeries performed chart ───────────────────────────────────────
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

    # ── Additional surgeries possible chart ───────────────────────────────────
    st.markdown("#### Additional Surgeries Possible from Recovered Schedule Time")
    st.caption(
        "Number of additional high-volume procedures that fit within OR time recovered "
        "on days where surgeries ran under their booked duration."
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

    # ── Breakdown by specialty ────────────────────────────────────────────────
    st.markdown("#### Specialty Breakdown")
    ref = summary[["ProcedureSpecialtyDescription",
                   "avg_procedure_duration_min", "total_additional_possible"]].copy()
    ref.columns = ["Specialty", "Avg Procedure Duration (min)", "Additional Surgeries Possible"]
    ref["Avg Procedure Duration (min)"] = ref["Avg Procedure Duration (min)"].round(1)
    st.dataframe(ref, use_container_width=True, hide_index=True)

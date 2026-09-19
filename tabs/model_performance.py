"""tabs/model_performance.py — Model Performance tab"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error


def render(model_stats, test_results):
    st.subheader("Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{model_stats['r2']:.3f}", help="Closer to 1.0 is better.")
    c2.metric("RMSE", f"{model_stats['rmse']:.1f} min", help="Root Mean Squared Error.")
    c3.metric("MAE", f"{model_stats['mae']:.1f} min", help="Mean Absolute Error.")
    c4.metric("Test Cases", f"{model_stats['test_size']:,}")

    st.divider()

    # ── Plot 1: Actual vs Predicted ───────────────────────────────────────────
    st.subheader("Actual vs. Predicted Duration")

    sample = test_results.sample(n=min(2000, len(test_results)), random_state=42)
    cap = float(np.percentile(test_results["actual"], 99))
    sample = sample[(sample["actual"] <= cap) & (sample["predicted"] <= cap)]

    fig1 = px.scatter(
        sample, x="actual", y="predicted",
        opacity=0.35,
        labels={"actual": "Actual Duration (min)", "predicted": "Predicted Duration (min)"},
        color_discrete_sequence=["#4A90D9"],
    )
    fig1.add_shape(
        type="line", x0=0, y0=0, x1=cap, y1=cap,
        line=dict(color="#E74C3C", width=2, dash="dash"),
    )
    fig1.add_annotation(
        x=cap * 0.85, y=cap * 0.92, text="Perfect prediction",
        showarrow=False, font=dict(color="#E74C3C", size=12),
    )
    fig1.update_layout(height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Red dashed line = perfect prediction. "
        "Points above the line are under-predicted (surgery ran longer than expected)."
    )

    st.divider()

    # ── Plot 2: Model vs Booked Duration ──────────────────────────────────────
    st.subheader("Model vs. Booked Duration")

    comparison = pd.DataFrame({
        "Source": ["Model Prediction", "Booked Duration"],
        "RMSE (min)": [model_stats["rmse"], model_stats["booked_rmse"]],
        "MAE (min)":  [model_stats["mae"],  model_stats["booked_mae"]],
    })

    col_rmse, col_mae = st.columns(2)

    with col_rmse:
        fig2a = px.bar(
            comparison, x="Source", y="RMSE (min)",
            color="Source", color_discrete_sequence=["#4A90D9", "#95A5A6"],
            text_auto=".1f",
        )
        fig2a.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig2a, use_container_width=True)

    with col_mae:
        fig2b = px.bar(
            comparison, x="Source", y="MAE (min)",
            color="Source", color_discrete_sequence=["#4A90D9", "#95A5A6"],
            text_auto=".1f",
        )
        fig2b.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig2b, use_container_width=True)

    rmse_imp = (1 - model_stats["rmse"] / model_stats["booked_rmse"]) * 100
    mae_imp  = (1 - model_stats["mae"]  / model_stats["booked_mae"])  * 100
    st.caption(
        f"The model reduces RMSE by **{rmse_imp:.1f}%** and MAE by **{mae_imp:.1f}%** "
        "compared to using the originally booked duration as a prediction."
    )

    st.divider()

    # ── Plot 3: Residuals Distribution ────────────────────────────────────────
    st.subheader("Residuals Distribution")

    residuals = test_results["residual"].clip(
        lower=float(np.percentile(test_results["residual"], 1)),
        upper=float(np.percentile(test_results["residual"], 99)),
    )
    fig3 = px.histogram(
        residuals, nbins=80,
        labels={"value": "Residual (Predicted - Actual, min)", "count": "Count"},
        color_discrete_sequence=["#4A90D9"],
    )
    fig3.add_vline(x=0, line_color="#E74C3C", line_dash="dash", line_width=2,
                   annotation_text="Zero error", annotation_position="top right")
    fig3.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    under = (test_results["residual"] < 0).mean() * 100
    over  = (test_results["residual"] > 0).mean() * 100
    st.caption(
        f"**{under:.1f}%** of cases are under-predicted (surgery ran longer than expected) · "
        f"**{over:.1f}%** are over-predicted."
    )

    st.divider()

    # ── Plot 4: MAE by Specialty ──────────────────────────────────────────────
    st.subheader("Prediction Error by Specialty")

    specialty_mae = (
        test_results.groupby("specialty")
        .apply(lambda g: mean_absolute_error(g["actual"], g["predicted"]))
        .reset_index()
        .rename(columns={0: "MAE (min)", "specialty": "Specialty"})
        .sort_values("MAE (min)", ascending=True)
    )

    fig4 = px.bar(
        specialty_mae, x="MAE (min)", y="Specialty",
        orientation="h",
        text_auto=".1f",
        color="MAE (min)",
        color_continuous_scale=["#4A90D9", "#E74C3C"],
    )
    fig4.update_layout(
        height=max(350, len(specialty_mae) * 28),
        margin=dict(t=20, b=20),
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig4, use_container_width=True)
    best  = specialty_mae.iloc[0]
    worst = specialty_mae.iloc[-1]
    st.caption(
        f"Best: **{best['Specialty']}** ({best['MAE (min)']:.1f} min MAE) · "
        f"Most challenging: **{worst['Specialty']}** ({worst['MAE (min)']:.1f} min MAE)"
    )

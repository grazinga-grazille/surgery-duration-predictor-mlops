"""tabs/prediction.py — Prediction tab"""

import streamlit as st


def render(run, description, surgical_priority, patient_type, room, specialty,
           predict_fn, fmt_duration_fn, model_stats):
    if not run:
        st.info(
            "Fill in the procedure details in the sidebar and click **Predict Duration**."
        )

        # Show sample stats so the page isn't blank
        st.subheader("Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cases", f"{model_stats['train_size'] + model_stats['test_size']:,}")
        c2.metric("Avg Duration", f"{model_stats['y_mean']:.0f} min")
        c3.metric("Median Duration", f"{model_stats['y_median']:.0f} min")
        c4.metric("Std Deviation", f"{model_stats['y_std']:.0f} min")

    else:
        if not description.strip():
            st.warning("Please enter a procedure description in the sidebar.")
        else:
            with st.spinner("Running prediction ..."):
                pred = predict_fn(surgical_priority, patient_type, room, specialty, description)

            st.markdown(
                f"""
                <div class="predict-box">
                    <p>Predicted Surgery Duration</p>
                    <h1>{pred:.0f} <span style="font-size:1.8rem">min</span></h1>
                    {f'<p style="font-size:1.1rem; margin-top:0.4rem; color:#ccd;">({fmt_duration_fn(pred)})</p>' if pred >= 60 else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

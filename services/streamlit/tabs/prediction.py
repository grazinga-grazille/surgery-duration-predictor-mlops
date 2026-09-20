"""tabs/prediction.py — Prediction tab"""

import streamlit as st


def render(run, description, surgical_priority, patient_type, room, specialty,
           predict_fn, fmt_duration_fn, model_stats):
    if not run:
        st.info(
            "Fill in the procedure details in the sidebar and click **Predict Duration**."
        )

        st.subheader("Dataset Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cases", f"{model_stats['train_size'] + model_stats['test_size']:,}")
        c2.metric("Avg Duration", f"{model_stats['y_mean']:.0f} min")
        c3.metric("Median Duration", f"{model_stats['y_median']:.0f} min")
        c4.metric("Std Deviation", f"{model_stats['y_std']:.0f} min")
        return

    if not description.strip():
        st.warning("Please enter a procedure description in the sidebar.")
        return

    with st.spinner("Calling API (embed + predict) ..."):
        try:
            result = predict_fn(
                surgical_priority, patient_type, room, specialty, description
            )
        except Exception as exc:
            st.error(str(exc))
            return

    if isinstance(result, tuple):
        pred, model_version = result
    else:
        pred, model_version = result, "api"

    # Keep the HTML block simple — Streamlit often escapes nested styled <p> tags.
    duration_line = ""
    if pred >= 60:
        duration_line = f"<p>({fmt_duration_fn(pred)})</p>"

    st.markdown(
        f"""
        <div class="predict-box">
            <p>Predicted Surgery Duration</p>
            <h1>{pred:.0f} <span style="font-size:1.8rem">min</span></h1>
            {duration_line}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"model={model_version}")

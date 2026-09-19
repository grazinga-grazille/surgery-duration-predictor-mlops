"""tabs/about.py — About tab"""

import streamlit as st


def render():
    st.subheader("About This Project")
    st.markdown(
        """
        **Surgery Duration Predictor** is a machine learning application developed as part of
        the research conducted in Department of Management Sciences at the **University of Waterloo**.

        ### Problem
        Inaccurate surgery time estimates lead to OR scheduling inefficiencies, patient wait
        times, and unnecessary costs. Most hospitals rely on surgeon-reported booked durations
        or historical averages which are often systematically biased.

        ### Approach
        A **Random Forest regression** model was trained on ~35,000 historical surgical cases
        from a Canadian teaching hospital. The pipeline combines:

        | Feature type | Details |
        |---|---|
        | Surgical priority | Numeric urgency score (1–5) |
        | Patient / room / specialty | One-hot encoded categorical features |
        | Procedure description | TF-IDF bigrams compressed to 150 SVD components |

        ### Target
        Actual OR time in minutes, **log-transformed** during training to reduce right-skew.
        Predictions are exponentiated back to minutes.

        ### Key Findings
        - The model outperforms booked duration benchmarks on test data.
        - Most outlier cases are **under-predicted** (surgery ran longer than expected).
        - Procedure description text (SVD embeddings) is the strongest predictor group.
        """
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a3a5c 0%, #0f2440 100%);
            border-left: 4px solid #e74c3c;
            border-radius: 8px;
            padding: 1rem 1.4rem;
            margin: 0.5rem 0 1.2rem 0;
        ">
            <p style="margin:0 0 0.3rem 0; font-size:0.78rem; color:#aac; letter-spacing:0.05em; text-transform:uppercase;">Key Contribution</p>
            <p style="margin:0; font-size:1rem; color:#f0f4f8; line-height:1.6;">
                Applied to 7,872 surgeries across 12 specialties (Mar 2024 – Mar 2025), replacing booked
                durations with model predictions reduces net OR scheduling cost by
                <strong style="color:#e74c3c;">$342,333</strong> — from $11.2M to $10.9M.
                Recovered schedule time creates capacity for
                <strong style="color:#e74c3c;">176 additional high-volume procedures</strong>
                without extending OR hours.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ---
        **Author:** Gatik Gola · University of Waterloo
        """
    )

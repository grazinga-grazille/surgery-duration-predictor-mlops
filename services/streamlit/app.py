"""app.py — Surgery Duration Predictor

Streamlit entry point: page config, styles, model loading, sidebar, tab wiring.
Tab content lives in tabs/*.py.

Prediction calls the FastAPI /predict endpoint (which loads the model from
MLflow artifacts and applies TF-IDF → SVD embeddings server-side). Local
models/*.pkl are only used for sidebar dropdowns / ML Analysis charts.
"""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request

import streamlit as st

from surgery_duration_predictor.artifacts import load_artifacts as _load_artifacts
from tabs import about, business_analysis, model_performance, prediction

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surgery Duration Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; color: #555; }
    .metric-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .predict-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .predict-box h1 { font-size: 3.5rem; margin: 0.2rem 0; }
    .predict-box p  { color: #aac; margin: 0; }
    /* Suggestion buttons in the sidebar — secondary buttons only */
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
        font-size: 0.7rem !important;
        text-align: left !important;
        border: 1px solid rgba(74, 144, 217, 0.35) !important;
        background: rgba(26, 58, 92, 0.55) !important;
        color: white !important;
        padding: 2px 8px !important;
        line-height: 1.3 !important;
    }
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
        background: #4A90D9 !important;
        color: white !important;
        border-color: #4A90D9 !important;
    }
    .ba-metric-card {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f2440 100%);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        border: 1px solid rgba(74, 144, 217, 0.25);
    }
    .ba-metric-card h3 { margin: 0 0 0.4rem 0; font-size: 0.85rem; color: #aac;
                         text-transform: uppercase; letter-spacing: 0.05em; }
    .ba-metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #f0f4f8; }
    .ba-drill-header {
        border-left: 4px solid #4A90D9;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
    }
    .ba-placeholder-note {
        background: rgba(74, 144, 217, 0.08);
        border: 1px dashed rgba(74, 144, 217, 0.35);
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #aac;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# ── Load artifacts (UI metadata only) ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading UI artifacts ...")
def load_artifacts():
    return _load_artifacts()


try:
    arts = load_artifacts()
    cat_values = arts["categorical_values"]
    valid_combinations = arts["valid_combinations"]
    procedure_examples = arts["procedure_examples"]
    model_stats = arts["model_stats"]
    test_results = arts["test_results"]
    patient_to_specialty = valid_combinations["patient_to_specialty"]
    patient_specialty_to_room = valid_combinations["patient_specialty_to_room"]
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def predict(surgical_priority, patient_type, room, specialty, description):
    """POST to FastAPI; server embeds procedure text (TF-IDF→SVD) and scores."""
    payload = {
        "surgical_priority": int(surgical_priority),
        "patient_type": patient_type,
        "room": room,
        "specialty": specialty,
        "procedure_description": description,
    }
    req = urllib.request.Request(
        f"{API_BASE_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach API at {API_BASE_URL}. "
            "Start FastAPI (uv run uvicorn … or docker compose up fastapi)."
        ) from exc
    return float(body["predicted_duration_minutes"]), str(
        body.get("model_version", "api")
    )


def fmt_duration(minutes: float) -> str:
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h}h {m:02d}m" if h > 0 else f"{m} min"


# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 Surgery Duration\nPredictor")
    st.caption("Department of Management Sciences · University of Waterloo")
    st.divider()

    if not model_loaded:
        st.error(
            "**UI artifacts not found.**\n\n"
            "Run `uv run scripts/train.py` once so sidebar dropdowns have values. "
            "Predictions still come from the FastAPI / MLflow model."
        )
        st.stop()

    st.subheader("Procedure Details")

    # ── Cascading dropdowns ───────────────────────────────────────────────────
    patient_type = st.selectbox("Patient Type", cat_values["PatientType"])

    valid_specialties = patient_to_specialty.get(
        patient_type, cat_values["ProcedureSpecialtyDescription"]
    )
    specialty = st.selectbox("Procedure Specialty", valid_specialties)

    valid_rooms = patient_specialty_to_room.get(
        (patient_type, specialty), cat_values["Roomdescription"]
    )
    room = st.selectbox("Operating Room", valid_rooms)

    surgical_priority = st.radio(
        "Surgical Priority",
        options=[1, 2, 3, 4, 5],
        index=1,
        horizontal=True,
        help="1 = most urgent, 5 = elective",
    )

    st.divider()

    # ── Procedure Recommendations ─────────────────────────────────────────────
    if "shuffle_seed" not in st.session_state:
        st.session_state.shuffle_seed = 0
    if "selected_proc" not in st.session_state:
        st.session_state.selected_proc = ""
    if "prev_specialty_rec" not in st.session_state:
        st.session_state.prev_specialty_rec = ""

    if st.session_state.prev_specialty_rec != specialty:
        st.session_state.shuffle_seed = 0
        st.session_state.selected_proc = ""
        st.session_state.prev_specialty_rec = specialty

    description = st.text_area(
        "Procedure Description",
        value=st.session_state.selected_proc,
        placeholder="Type a description or pick a suggestion below",
        height=90,
    )

    sug_col, btn_col = st.columns([3, 1])
    sug_col.caption("Suggestions")
    if btn_col.button("🔀", help="Shuffle suggestions", use_container_width=True):
        st.session_state.shuffle_seed += 1
        st.session_state.selected_proc = ""
        st.rerun()

    pool = procedure_examples.get(specialty, [])
    rng = random.Random(st.session_state.shuffle_seed)
    shown = rng.sample(pool, min(5, len(pool))) if pool else []

    for proc in shown:
        display = proc if len(proc) <= 42 else proc[:40] + "..."
        display = display[0].upper() + display[1:] if display else display
        if st.button(display, key=f"rec_{proc}", use_container_width=True):
            st.session_state.selected_proc = proc
            st.rerun()

    st.divider()
    run = st.button("Predict Duration", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
tab_predict, tab_model, tab_business, tab_about = st.tabs(
    ["🔮 Prediction", "📊 ML Analysis", "💼 Business Analysis", "ℹ️ About"]
)

with tab_predict:
    prediction.render(
        run, description, surgical_priority, patient_type, room, specialty,
        predict_fn=predict,
        fmt_duration_fn=fmt_duration,
        model_stats=model_stats,
    )

with tab_model:
    model_performance.render(model_stats, test_results)

with tab_business:
    business_analysis.render()

with tab_about:
    about.render()

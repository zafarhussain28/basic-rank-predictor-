from typing import Any

import streamlit as st

from predictor import (
    APP_DIR,
    DEFAULT_RAW_MARKS,
    MAX_RAW_MARKS,
    MIN_RAW_MARKS,
    _load_models,
    predict_scores,
)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Any, Any]:
    return _load_models(APP_DIR)


st.set_page_config(
    page_title="JEE Rank & Percentile Predictor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #f3f7fb 0%, #eef5f8 55%, #f7f8fb 100%);
        color: #17324a;
    }
    [data-testid="stHeader"] { background: transparent; }
    .main .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] {
        color: #263f54;
    }
    [data-testid="stWidgetLabel"] p { font-weight: 600; }
    [data-testid="stCaptionContainer"] { color: #536b80; }
    .hero {
        padding: 2rem 2.2rem;
        background: linear-gradient(115deg, #10243e, #174f78);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 8px;
        color: #f7fbff;
        margin-bottom: 1.4rem;
        box-shadow: 0 14px 34px rgba(16, 36, 62, .12);
    }
    .hero-kicker { color: #8fe1d0; font-size: .78rem; font-weight: 700; text-transform: uppercase; }
    .hero h1 { color: #fff; font-size: 2rem; line-height: 1.18; margin: .55rem 0 .45rem; }
    .hero p { color: #d3e2ef; font-size: 1rem; margin: 0; max-width: 700px; }
    .notice {
        color: #664a10;
        background: #fff8e7;
        border-left: 4px solid #e6b84f;
        border-radius: 4px;
        padding: .8rem 1rem;
        margin: 0 0 1.4rem;
        font-size: .92rem;
    }
    .result-label { color: #d6e7f2; font-size: .88rem; margin-bottom: .35rem; }
    .result-value { color: #fff; font-size: 2rem; line-height: 1.2; font-weight: 700; }
    .result-unit { color: #b7d0df; font-size: .83rem; margin-top: .35rem; }
    .result-card {
        min-height: 145px;
        background: #123553;
        border: 1px solid #2e6385;
        border-left: 4px solid #5ed0b5;
        border-radius: 8px;
        padding: 1.1rem;
        box-shadow: 0 10px 24px rgba(16, 36, 62, .10);
    }
    .result-card.rank { background: #155469; border-color: #3e8991; border-left-color: #f0c66a; }
    .section-note { color: #526579; font-size: .91rem; }
    [data-testid="stNumberInput"] input {
        color: #17324a !important;
        background: #ffffff !important;
        border: 1px solid #c5d3df !important;
        border-radius: 6px;
    }
    [data-testid="stNumberInput"] button {
        color: #34566d !important;
        background: #f7fafc !important;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #087c78;
        border: 1px solid #087c78;
        color: #ffffff;
        min-height: 2.8rem;
        border-radius: 6px;
        font-weight: 700;
        box-shadow: 0 5px 12px rgba(8, 124, 120, .18);
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #066763;
        border-color: #066763;
    }
    [data-testid="stFormSubmitButton"] button:focus-visible,
    [data-testid="stNumberInput"] input:focus-visible {
        outline: 3px solid rgba(8, 124, 120, .26);
        outline-offset: 2px;
    }
    @media (max-width: 640px) {
        .main .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.4rem; }
        .hero h1 { font-size: 1.55rem; }
        .result-value { font-size: 1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="hero">
      <div class="hero-kicker">JEE Main estimate tool</div>
      <h1>JEE Rank &amp; Percentile Predictor</h1>
      <p>Explore a model-based estimate from your raw marks using the project's trained prediction models.</p>
    </header>
    <div class="notice"><strong>Estimate only.</strong> Results are not official examination scores, ranks, or counselling guidance.</div>
    """,
    unsafe_allow_html=True,
)

input_column, result_column = st.columns([0.9, 1.1], gap="large")

with input_column:
    st.subheader("Your marks")
    st.caption("Enter one raw marks value. Both trained models use this same input.")
    with st.form("prediction_form"):
        raw_marks = st.number_input(
            "Raw marks (out of 300)",
            min_value=MIN_RAW_MARKS,
            max_value=MAX_RAW_MARKS,
            value=DEFAULT_RAW_MARKS,
            step=1,
            help="The training dataset contains integer raw marks from 0 through 300.",
        )
        submitted = st.form_submit_button(
            "Predict percentile and rank", type="primary", use_container_width=True
        )
    st.markdown(
        '<p class="section-note">The models were trained on the project dataset and should only be interpreted within its marks range.</p>',
        unsafe_allow_html=True,
    )

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

if submitted:
    st.session_state.prediction_result = None
    st.session_state.prediction_error = None
    try:
        with st.spinner("Loading trained models and calculating your estimate..."):
            percentile_model, rank_model = load_models()
            percentile, rank = predict_scores(raw_marks, percentile_model, rank_model)
        st.session_state.prediction_result = (int(raw_marks), percentile, rank)
    except Exception as error:
        st.session_state.prediction_error = str(error)

with result_column:
    st.subheader("Prediction results")
    result = st.session_state.prediction_result
    if result:
        predicted_marks, percentile, rank = result
        percentile_text = f"{percentile:,.5f}"
        rank_text = f"{rank:,.0f}"
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Predicted percentile</div>
              <div class="result-value">{percentile_text}</div>
              <div class="result-unit">percentile points</div>
            </div>
            <br>
            <div class="result-card rank">
              <div class="result-label">Predicted rank</div>
              <div class="result-value">{rank_text}</div>
              <div class="result-unit">approximate rank</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Input used: {predicted_marks} / 300 raw marks")
    elif st.session_state.prediction_error:
        st.error(f"Prediction could not be completed: {st.session_state.prediction_error}")
    else:
        st.info("Your estimates will appear here after you submit your marks.")

st.divider()
st.subheader("How it works")
st.write(
    "Two separately trained Random Forest regression models map `raw_marks` to "
    "`approx_percentile` and `approx_rank`. The serving app uses the original "
    "feature name and a one-column numeric input for both models."
)
st.markdown(
    '<p class="section-note">The supplied dataset contains 301 rows of approximate targets. Model estimates can differ from official results and are not a guarantee of rank, percentile, admission, or eligibility.</p>',
    unsafe_allow_html=True,
)
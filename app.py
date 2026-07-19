from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
HOUSE_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "house_price_best_model.joblib"
TELCO_MODEL_PATH = ROOT_DIR / "outputs" / "models" / "telco_churn_best_model.joblib"


st.set_page_config(page_title="ML Business Predictions", layout="wide")


@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)


def missing_model_message(path: Path) -> None:
    st.warning(f"Model file not found: {path}")
    st.code("python -m src.train", language="bash")


def render_input_form(artifact: dict, form_key: str) -> pd.DataFrame | None:
    metadata = artifact["metadata"]
    schema = metadata["feature_schema"]
    features = metadata["features"]

    with st.form(form_key):
        values = {}
        columns = st.columns(2)
        for idx, feature in enumerate(features):
            field = schema[feature]
            with columns[idx % 2]:
                if field["kind"] == "numeric":
                    value = st.number_input(
                        feature,
                        value=float(field["default"]),
                        format="%.4f",
                    )
                else:
                    choices = field.get("choices", [""])
                    default = field.get("default", choices[0])
                    default_index = choices.index(default) if default in choices else 0
                    value = st.selectbox(feature, choices, index=default_index)
                values[feature] = value

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return None

    return pd.DataFrame([values], columns=features)


def render_regression_tab() -> None:
    st.subheader("House price prediction")
    if not HOUSE_MODEL_PATH.exists():
        missing_model_message(HOUSE_MODEL_PATH)
        return

    artifact = load_artifact(HOUSE_MODEL_PATH)
    st.caption(f"Model: {artifact['metadata']['model_name']}")
    X = render_input_form(artifact, "house_form")
    if X is None:
        return

    prediction = float(artifact["pipeline"].predict(X)[0])
    st.metric("Predicted sale price", f"${prediction:,.0f}")


def render_classification_tab() -> None:
    st.subheader("Customer churn prediction")
    if not TELCO_MODEL_PATH.exists():
        missing_model_message(TELCO_MODEL_PATH)
        return

    artifact = load_artifact(TELCO_MODEL_PATH)
    st.caption(f"Model: {artifact['metadata']['model_name']}")
    X = render_input_form(artifact, "telco_form")
    if X is None:
        return

    pipeline = artifact["pipeline"]
    label_map = artifact["metadata"].get("labels", {0: "No", 1: "Yes"})
    prediction = int(pipeline.predict(X)[0])
    prediction_label = label_map.get(prediction, str(prediction))

    if hasattr(pipeline, "predict_proba"):
        churn_probability = float(pipeline.predict_proba(X)[0][1])
        st.metric("Churn risk", f"{churn_probability * 100:.1f}%")

    st.metric("Predicted class", prediction_label)


st.title("ML Business Predictions")
st.write(
    "Minimal GUI for the Machine Learning project: house price regression and telecom churn classification."
)

regression_tab, classification_tab = st.tabs(
    ["House Price Regression", "Telco Churn Classification"]
)

with regression_tab:
    render_regression_tab()

with classification_tab:
    render_classification_tab()

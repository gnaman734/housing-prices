from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.io_utils import load_artifacts, save_artifacts
from src.modeling import prepare_inference_data, train_model

st.set_page_config(page_title="Intelligent Property Price Prediction", layout="wide")

MODEL_CHOICES = {
    "Linear Regression": "linear_regression",
    "Decision Tree Regressor": "decision_tree",
    "Random Forest Regressor": "random_forest",
}


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _display_metrics(metrics: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", _format_currency(metrics["MAE"]))
    c2.metric("RMSE", _format_currency(metrics["RMSE"]))
    c3.metric("R2", f"{metrics['R2']:.4f}")


st.title("Project 9: Intelligent Property Price Prediction and Agentic Real Estate Advisory System")
st.caption("Milestone 1: ML-Based Property Price Prediction")

with st.sidebar:
    st.header("Train Model")
    uploaded_train = st.file_uploader("Upload historical property dataset (CSV)", type=["csv"])
    selected_model_label = st.selectbox("Model", options=list(MODEL_CHOICES.keys()), index=2)
    test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

if "trained_pipeline" not in st.session_state:
    st.session_state.trained_pipeline = None
    st.session_state.metadata = None

train_tab, predict_tab, advisory_tab = st.tabs(["Model Training", "Price Prediction", "Advisory Summary"])

with train_tab:
    st.subheader("1) Upload and Train")
    if not uploaded_train:
        st.info("Upload a CSV to begin. Example file: data/sample_property_data.csv")
    else:
        df_train = pd.read_csv(uploaded_train)
        st.write("Dataset preview", df_train.head())

        candidate_targets = df_train.columns.tolist()
        default_target_idx = candidate_targets.index("price") if "price" in candidate_targets else 0
        target_col = st.selectbox("Target column (property price)", options=candidate_targets, index=default_target_idx)

        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                result = train_model(
                    df=df_train,
                    target_col=target_col,
                    model_name=MODEL_CHOICES[selected_model_label],
                    test_size=test_size,
                )

            st.session_state.trained_pipeline = result.pipeline
            st.session_state.metadata = {
                "feature_columns": result.feature_columns,
                "target_column": target_col,
                "model_name": MODEL_CHOICES[selected_model_label],
                "metrics": result.metrics,
            }

            st.success("Training complete")
            _display_metrics(result.metrics)

            st.markdown("### Key Price-Driving Features")
            if result.feature_importance.empty:
                st.write("Feature importance is not available for this model.")
            else:
                chart_df = result.feature_importance.set_index("feature")
                st.bar_chart(chart_df)
                st.dataframe(result.feature_importance, use_container_width=True)

            model_path, meta_path = save_artifacts(
                pipeline=result.pipeline,
                feature_columns=result.feature_columns,
                target_column=target_col,
                model_name=MODEL_CHOICES[selected_model_label],
                metrics=result.metrics,
            )
            st.caption(f"Saved artifacts: {model_path}, {meta_path}")

    st.divider()
    st.subheader("Load Existing Artifacts")
    if st.button("Load model from /models"):
        try:
            pipeline, metadata = load_artifacts()
            st.session_state.trained_pipeline = pipeline
            st.session_state.metadata = metadata
            st.success("Model loaded from models directory")
            _display_metrics(metadata.get("metrics", {"MAE": 0, "RMSE": 0, "R2": 0}))
        except Exception as exc:
            st.error(f"Unable to load artifacts: {exc}")

with predict_tab:
    st.subheader("2) Batch Price Prediction")
    uploaded_pred = st.file_uploader("Upload properties to predict (CSV)", type=["csv"], key="pred_uploader")

    if st.session_state.trained_pipeline is None:
        st.warning("Train a model first or load existing artifacts from /models.")
    elif not uploaded_pred:
        st.info("Upload a CSV of new properties for prediction.")
    else:
        input_df = pd.read_csv(uploaded_pred)
        st.write("Input preview", input_df.head())

        prepared = prepare_inference_data(input_df, st.session_state.metadata["feature_columns"])
        preds = st.session_state.trained_pipeline.predict(prepared)

        output_df = input_df.copy()
        output_df["predicted_price"] = preds
        st.dataframe(output_df, use_container_width=True)

        csv_data = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions CSV",
            data=csv_data,
            file_name="predicted_property_prices.csv",
            mime="text/csv",
        )

with advisory_tab:
    st.subheader("3) Basic Real Estate Advisory (Milestone 1 Starter)")
    st.write(
        "This view produces a structured recommendation based on model outputs. "
        "In Milestone 2, this can be upgraded to an agentic workflow using LangGraph + an open-source LLM."
    )

    if st.session_state.trained_pipeline is None:
        st.info("Train or load a model to enable advisory insights.")
    else:
        metrics = st.session_state.metadata.get("metrics", {})
        quality_note = "High confidence" if metrics.get("R2", 0) >= 0.75 else "Moderate confidence"
        st.markdown(
            f"""
            **Recommendation Template**

            - **Model quality**: {quality_note} (R2 = {metrics.get('R2', 0):.3f})
            - **Suggested use**: shortlist filtering and initial valuation checks
            - **Risk flag**: always validate with local market comps and legal checks
            - **Investor action**: prioritize properties where predicted value materially exceeds listing price after renovation/holding costs
            """
        )

st.divider()
st.caption("Free-tier stack: Streamlit + scikit-learn + pandas. Ready for Streamlit Community Cloud or Hugging Face Spaces.")

if not Path("data/sample_property_data.csv").exists():
    st.warning("Sample file missing at data/sample_property_data.csv")

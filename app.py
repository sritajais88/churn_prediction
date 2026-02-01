"""
Streamlit app for Customer Churn Prediction.

Note:
- To avoid "attempted relative import" errors when Streamlit runs the script,
  this file inserts the project root into sys.path at runtime (local dev convenience).
"""

# Shim: make project root importable (helps when Streamlit changes cwd)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import DATA_FILE, TARGET_COL
from src.deployment import predict_single, predict_batch
from src.preprocessing import data_load

# Cache loading sample data to build UI
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = data_load(DATA_FILE)
    return df


def build_input_form(df_sample: pd.DataFrame) -> Dict[str, Any]:
    """
    Build input widgets based on df_sample columns.
    Special-case: SeniorCitizen is presented as a categorical (0/1).
    """
    st.subheader("Enter Customer Details")
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]

    input_data: Dict[str, Any] = {}
    for col in feature_cols:
        series = df_sample[col]

        # Special-case for SeniorCitizen: show selectbox with 0/1
        if col == "SeniorCitizen":
            options = sorted(series.dropna().unique().astype(int).tolist()) if not series.dropna().empty else [0, 1]
            # ensure options are 0/1
            options = [int(o) for o in options] if options else [0, 1]
            input_data[col] = st.selectbox(label=col, options=options, index=0)
            continue

        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.isna().all() else 0.0
            input_data[col] = st.number_input(label=col, value=default)
        else:
            options = series.dropna().unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(label=col, options=options, index=0)
    return input_data


def main() -> None:
    st.title("Customer Churn Prediction")

    st.markdown(
        """
        Input customer information and predict whether the customer will churn.
        The model uses preprocessing pipelines (imputer + scaler, and imputer + OHE).
        """
    )

    df_sample = load_sample_data()
    st.sidebar.markdown("### Data snapshot")
    st.sidebar.dataframe(df_sample.head(5))

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        input_data = build_input_form(df_sample)
        st.write("Input preview:")
        st.json(input_data)

        if st.button("Predict Churn"):
            try:
                result = predict_single(input_data)
                churn_prob = result["churn_probability"]
                st.write("**Predicted class:**", result["prediction"])
                st.write("**Churn probability:**", f"{churn_prob:.4f}")
                if churn_prob > 0.5:
                    st.error("High risk of churn")
                else:
                    st.success("Low risk of churn")
            except Exception as exc:
                st.error("Prediction failed — check logs for details.")
                st.write(f"Error: {str(exc)}")

    with tab2:
        st.subheader("Upload a cleaned CSV for batch predictions (must contain same columns as training)")
        uploaded = st.file_uploader("Choose a CSV", type=["csv"])
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(uploaded_df.head())
            if st.button("Run batch prediction"):
                try:
                    result_df = predict_batch(uploaded_df)
                    st.write("Prediction results (first 10 rows):")
                    st.dataframe(result_df.head(10))
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")
                except Exception as exc:
                    st.error("Batch prediction failed — check logs for details.")
                    st.write(f"Error: {str(exc)}")


if __name__ == "__main__":
    main()

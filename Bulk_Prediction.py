
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Bulk Churn Prediction",
    page_icon="📁",
    layout="wide"
)

st.title("Bulk Customer Churn Prediction")
st.write("Upload a CSV file to predict churn for multiple customers.")

project_dir = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model():
    return joblib.load(project_dir / "churn_prediction_model.joblib")

package = load_model()

model = package["pipeline"]
threshold = package["threshold"]
features = package["feature_columns"]
mappings = package["category_mappings"]

csv_files = list(project_dir.glob("ecommerce_churn_cleaned*.csv"))

if csv_files:
    template_df = pd.read_csv(csv_files[0])
    template = template_df[features].head(5).to_csv(index=False)

    st.download_button(
        "Download Sample CSV Template",
        data=template,
        file_name="customer_template.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        missing_columns = [
            col for col in features
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        st.success(
            f"Successfully loaded {len(df):,} customer records!"
        )

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button(
            "Predict Churn for All Customers",
            type="primary"
        ):

            X_bulk = df[features].copy()

            for col, mapping in mappings.items():
                X_bulk[col] = X_bulk[col].replace(mapping)

            X_bulk["Complain"] = X_bulk["Complain"].replace({
                "Yes": 1,
                "No": 0,
                "yes": 1,
                "no": 0
            })

            categorical_columns = [
                "PreferredLoginDevice",
                "PreferredPaymentMode",
                "Gender",
                "PreferedOrderCat",
                "MaritalStatus"
            ]

            for col in categorical_columns:
                X_bulk[col] = X_bulk[col].map(
                    lambda x: x.strip()
                    if isinstance(x, str) else x
                )

            numeric_columns = [
                col for col in features
                if col not in categorical_columns
            ]

            for col in numeric_columns:
                X_bulk[col] = pd.to_numeric(
                    X_bulk[col],
                    errors="coerce"
                )

            probabilities = model.predict_proba(
                X_bulk
            )[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(int)

            output = df.copy()

            output["Churn_Score"] = np.round(
                probabilities, 4
            )

            output["Predicted_Churn"] = predictions

            output["Risk_Category"] = np.where(
                predictions == 1,
                "At Risk",
                "Predicted to Stay"
            )

            st.divider()
            st.subheader("Prediction Results")

            total = len(output)
            at_risk = int(predictions.sum())
            retained = total - at_risk

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Customers", f"{total:,}")
            c2.metric("At-Risk Customers", f"{at_risk:,}")
            c3.metric("Predicted to Stay", f"{retained:,}")

            st.dataframe(
                output,
                use_container_width=True
            )

            csv = output.to_csv(index=False)

            st.download_button(
                "Download Prediction Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.caption(
                "Predictions use the saved Random Forest "
                f"and a classification threshold of {threshold:.2f}. "
                "Model scores are not guaranteed to be "
                "calibrated churn probabilities."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Analytics Dashboard")
st.write("Explore customer behavior and churn patterns.")

# Find our dataset
project_dir = Path(__file__).resolve().parent.parent
csv_files = list(
    project_dir.glob("ecommerce_churn_cleaned*.csv")
)

if not csv_files:
    st.error("Cleaned dataset not found.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(csv_files[0])

# Main statistics
total = len(df)
churned = int(df["Churn"].sum())
retained = total - churned
churn_rate = (churned / total) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total:,}")
c2.metric("Retained Customers", f"{retained:,}")
c3.metric("Churned Customers", f"{churned:,}")
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# Satisfaction and complaints
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn by Satisfaction")

    satisfaction = (
        df.groupby("SatisfactionScore")["Churn"]
        .mean()
        .mul(100)
    )

    st.bar_chart(satisfaction)

with col2:
    st.subheader("Churn by Complaints")

    complaints = (
        df.groupby("Complain")["Churn"]
        .mean()
        .mul(100)
    )

    complaints.index = ["No Complaint", "Complaint"]

    st.bar_chart(complaints)

st.divider()

# Product category analysis
st.subheader("Churn by Product Category")

category_churn = (
    df.groupby("PreferedOrderCat")["Churn"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

st.bar_chart(category_churn)

st.caption(
    "All charts show historical churn percentages "
    "within each customer group."
)

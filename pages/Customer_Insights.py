
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Customer Insights",
    page_icon="🔍",
    layout="wide"
)

st.title("Customer Churn Insights")
st.write(
    "Understand which customer features "
    "influence our Random Forest predictions."
)

# Actual permutation importance results
importance_data = {
    "Feature": [
        "Tenure",
        "Complain",
        "NumberOfAddress",
        "CashbackAmount",
        "SatisfactionScore",
        "PreferedOrderCat",
        "DaySinceLastOrder",
        "WarehouseToHome",
        "MaritalStatus",
        "PreferredPaymentMode",
        "OrderAmountHikeFromlastYear",
        "CityTier",
        "NumberOfDeviceRegistered",
        "PreferredLoginDevice",
        "OrderCount",
        "CouponUsed",
        "HourSpendOnApp",
        "Gender"
    ],
    "Importance": [
        0.2078, 0.0673, 0.0292,
        0.0223, 0.0163, 0.0126,
        0.0090, 0.0090, 0.0084,
        0.0076, 0.0067, 0.0065,
        0.0038, 0.0038, 0.0032,
        0.0020, 0.0017, 0.0008
    ]
}

importance_df = pd.DataFrame(importance_data)

# Top feature cards
st.subheader("Most Influential Features")

c1, c2, c3 = st.columns(3)

c1.metric("Top Predictor", "Tenure")
c2.metric("Second Predictor", "Complaints")
c3.metric("Third Predictor", "Number of Addresses")

st.divider()

# Feature importance chart
st.subheader("Feature Importance Ranking")

chart = alt.Chart(importance_df).mark_bar(
    color="#38bdf8"
).encode(
    x=alt.X(
        "Importance:Q",
        title="Mean Decrease in ROC-AUC"
    ),
    y=alt.Y(
        "Feature:N",
        sort="-x",
        title=None
    ),
    tooltip=[
        "Feature",
        alt.Tooltip("Importance:Q", format=".4f")
    ]
).properties(height=550)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Permutation importance measures how much "
    "ROC-AUC decreases when a feature is shuffled. "
    "It does not measure causal effects."
)

st.divider()

# Key customer insights
st.subheader("Key Findings")

with st.expander("1. Customer Tenure", expanded=True):
    st.write(
        "Tenure is the strongest predictor in our model. "
        "Customers with shorter tenure generally have "
        "higher churn risk."
    )

with st.expander("2. Customer Complaints"):
    st.write(
        "Customers who made complaints showed higher "
        "historical churn rates. Complaints are the "
        "second-most influential predictor."
    )

with st.expander("3. Number of Addresses"):
    st.write(
        "The number of customer addresses is the third "
        "most influential feature. Its relationship "
        "with churn requires further investigation."
    )

with st.expander("4. Cashback Amount"):
    st.write(
        "Higher cashback is associated with lower churn "
        "overall. However, tenure and product category "
        "may explain part of this relationship."
    )

with st.expander("5. Satisfaction Score"):
    st.write(
        "Higher satisfaction scores unexpectedly showed "
        "higher historical churn rates in this dataset. "
        "The reason for this relationship remains unclear."
    )

st.info(
    "These findings describe patterns learned from "
    "historical data. They do not prove that changing "
    "any particular feature will prevent churn."
)

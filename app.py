
import streamlit as st
import joblib

# Configure application
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("E-Commerce Customer Churn Prediction")

st.write(
    "Machine Learning Based Customer Retention System"
)

# Load trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_model.joblib")

model_package = load_model()

# Extract saved model information
model = model_package["pipeline"]
threshold = model_package["threshold"]
features = model_package["feature_columns"]

# Display model status
st.success("Machine Learning Model Loaded Successfully!")

st.write("**Algorithm:** Random Forest")
st.write("**Classification Threshold:**", threshold)
st.write("**Number of Input Features:**", len(features))

st.info(
    "The application is ready for customer prediction integration."
)



import pandas as pd

st.divider()
st.header("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk.")

# Create customer input form
with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Behavior")

        tenure = st.number_input(
            "Tenure (Months)", min_value=0, value=6
        )

        satisfaction = st.selectbox(
            "Satisfaction Score", [1, 2, 3, 4, 5],
            index=2
        )

        complain = st.selectbox(
            "Has Customer Complained?", ["No", "Yes"]
        )

        cashback = st.number_input(
            "Cashback Amount", min_value=0.0,
            value=180.0
        )

        order_count = st.number_input(
            "Number of Orders", min_value=0, value=3
        )

        days_since_order = st.number_input(
            "Days Since Last Order",
            min_value=0, value=5
        )

        coupon_used = st.number_input(
            "Coupons Used", min_value=0, value=1
        )

        app_hours = st.number_input(
            "Hours Spent on App", min_value=0.0,
            value=3.0
        )

        order_hike = st.number_input(
            "Order Amount Hike From Last Year",
            min_value=0.0, value=15.0
        )

    with col2:
        st.subheader("Customer Profile")

        order_category = st.selectbox(
            "Preferred Order Category",
            ["Mobile Phone", "Laptop & Accessory",
             "Fashion", "Grocery", "Others"]
        )

        login_device = st.selectbox(
            "Preferred Login Device",
            ["Mobile Phone", "Computer"]
        )

        payment = st.selectbox(
            "Preferred Payment Method",
            ["Debit Card", "Credit Card", "E wallet",
             "Cash on Delivery", "UPI"]
        )

        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        city_tier = st.selectbox(
            "City Tier", [1, 2, 3]
        )

        warehouse_distance = st.number_input(
            "Warehouse To Home Distance",
            min_value=0.0, value=15.0
        )

        devices = st.number_input(
            "Number of Registered Devices",
            min_value=1, value=3
        )

        addresses = st.number_input(
            "Number of Addresses",
            min_value=1, value=2
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# Make prediction when button is clicked
if submitted:

    customer_data = {
        "Tenure": tenure,
        "PreferredLoginDevice": login_device,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_distance,
        "PreferredPaymentMode": payment,
        "Gender": gender,
        "HourSpendOnApp": app_hours,
        "NumberOfDeviceRegistered": devices,
        "PreferedOrderCat": order_category,
        "SatisfactionScore": satisfaction,
        "MaritalStatus": marital_status,
        "NumberOfAddress": addresses,
        "Complain": 1 if complain == "Yes" else 0,
        "OrderAmountHikeFromlastYear": order_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_since_order,
        "CashbackAmount": cashback
    }

    # Convert inputs into a dataframe
    customer_df = pd.DataFrame([customer_data])

    # Match the exact feature order used in training
    customer_df = customer_df[features]

    # Predict customer churn score
    churn_score = model.predict_proba(
        customer_df
    )[0, 1]

    prediction = int(churn_score >= threshold)

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        "Model Churn Score",
        f"{churn_score * 100:.2f}%"
    )

    if prediction == 1:
        st.error(
            "HIGH CHURN RISK - Customer flagged as at risk."
        )
    else:
        st.success(
            "LOW CHURN RISK - Customer predicted to stay."
        )

    st.caption(
        f"Classification threshold: {threshold:.0%}. "
        "This model score is not necessarily a "
        "calibrated real-world churn probability."
    )

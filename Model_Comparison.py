
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🏆",
    layout="wide"
)

st.title("Machine Learning Model Comparison")
st.write(
    "Performance comparison of our five "
    "classification algorithms."
)

# Results from our group-aware cross-validation
results = {
    "Model": [
        "Random Forest",
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "Original Logistic Regression",
        "Balanced Logistic Regression"
    ],
    "Accuracy": [
        0.9368, 0.9319, 0.9242,
        0.9204, 0.8887, 0.8000
    ],
    "Precision": [
        0.9164, 0.8652, 0.8140,
        0.8346, 0.7527, 0.4496
    ],
    "Recall": [
        0.6886, 0.7031, 0.7098,
        0.6578, 0.5126, 0.8030
    ],
    "F1-score": [
        0.7858, 0.7741, 0.7578,
        0.7341, 0.6075, 0.5759
    ],
    "ROC-AUC": [
        0.9791, 0.9688, 0.9596,
        0.9554, 0.8857, 0.8841
    ],
    "PR-AUC": [
        0.9150, 0.8874, 0.8575,
        0.8482, 0.7031, 0.6885
    ]
}

df_results = pd.DataFrame(results)

# Convert scores to percentages for display
display_df = df_results.copy()
metric_cols = display_df.columns.drop("Model")
display_df[metric_cols] = (
    display_df[metric_cols] * 100
).round(2)

st.subheader("Cross-Validation Results")
st.caption(
    "Mean results from group-aware 5-fold "
    "cross-validation on the training data."
)

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True
)

# F1-score comparison chart

# F1-score comparison
st.subheader("Model F1-Score Comparison")

f1_chart = alt.Chart(display_df).mark_bar(
    color="#38bdf8",
    cornerRadiusEnd=5
).encode(
    x=alt.X(
        "F1-score:Q",
        title="F1-score (%)",
        scale=alt.Scale(domain=[0, 100])
    ),
    y=alt.Y(
        "Model:N",
        sort="-x",
        title=None
    ),
    tooltip=["Model", "F1-score"]
).properties(height=300)

st.altair_chart(f1_chart, use_container_width=True)


# Precision vs Recall
st.subheader("Precision vs Recall")

pr_df = display_df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall"],
    var_name="Metric",
    value_name="Score"
)

pr_chart = alt.Chart(pr_df).mark_bar().encode(
    x=alt.X(
        "Model:N",
        title=None,
        axis=alt.Axis(labelAngle=-30)
    ),
    y=alt.Y(
        "Score:Q",
        title="Score (%)",
        scale=alt.Scale(domain=[0, 100])
    ),
    xOffset="Metric:N",
    color="Metric:N",
    tooltip=["Model", "Metric", "Score"]
).properties(height=350)

st.altair_chart(pr_chart, use_container_width=True)


# Precision and recall comparison
st.subheader("Precision vs Recall")

pr_data = display_df.set_index("Model")[
    ["Precision", "Recall"]
]
st.bar_chart(pr_data)

st.divider()

# Final model information
st.subheader("Selected Model: Tuned Random Forest")

st.write(
    "Random Forest was selected because it "
    "achieved the best overall F1-score and "
    "PR-AUC in our initial group-aware comparison."
)

st.write("**Selected Classification Threshold:** 0.40")
st.write("**Tuned Model Selection CV F1:** 79.37%")

st.subheader("Group-Aware Test Results")

# Results calculated from our final confusion matrix
tn, fp, fn, tp = 921, 19, 22, 160

accuracy = (tn + tp) / (tn + fp + fn + tp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1-score", f"{f1:.2%}")

st.info(
    "Test results use a group-aware split "
    "with no identical feature profiles shared "
    "between training and testing."
)

st.caption(
    "This is a research evaluation on a historical "
    "dataset, not independent validation on live "
    "e-commerce customers. An earlier test partition "
    "was examined during development."
)

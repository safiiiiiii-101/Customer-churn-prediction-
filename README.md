
# E-Commerce Customer Churn Prediction Using Machine Learning

## Project Overview

This project develops a machine-learning system to predict
customer churn in an e-commerce business.

The system analyzes customer behavior, purchasing patterns,
complaints, satisfaction scores, and other characteristics
to identify customers who may stop using the platform.

An interactive Streamlit application allows users to explore
customer data, compare machine-learning models, understand
important prediction features, and predict churn for both
individual customers and uploaded CSV datasets.

## Dataset

Dataset: E-Commerce Customer Churn
Source: Kaggle

- Total customers: 5,630
- Original columns: 20
- Prediction target: Churn
- Model input features: 18

Target labels:
- 0: Customer retained
- 1: Customer churned

Historical churn distribution:
- Retained customers: 4,682
- Churned customers: 948
- Churn rate: 16.84%

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Streamlit
- Altair
- Joblib
- Google Colab

## Project Workflow

1. Dataset exploration and statistical analysis
2. Missing-value handling and category standardization
3. Data visualization and churn analysis
4. Numerical scaling and categorical encoding
5. Training multiple classification algorithms
6. Group-aware cross-validation
7. Random Forest hyperparameter optimization
8. Classification threshold selection
9. Model evaluation
10. Permutation importance and SHAP analysis
11. Streamlit application development
12. Individual and bulk customer predictions

## Machine Learning Models

The following algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Random Forest achieved the strongest F1-score in the
initial group-aware cross-validation comparison.

### Group-Aware Cross-Validation Results

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 93.68% | 78.58% | 97.91% |
| CatBoost | 93.19% | 77.41% | 96.88% |
| LightGBM | 92.42% | 75.78% | 95.96% |
| XGBoost | 92.04% | 73.41% | 95.54% |
| Original Logistic Regression | 88.87% | 60.75% | 88.57% |
| Balanced Logistic Regression | 80.00% | 57.59% | 88.41% |

## Selected Model

Algorithm: Tuned Random Forest

Classification threshold: 0.40

Final group-aware test results:

- Accuracy: 96.35%
- Precision: 89.39%
- Recall: 87.91%
- F1-score: 88.64%

Confusion matrix:

| | Predicted Retained | Predicted Churn |
|---|---:|---:|
| Actual Retained | 921 | 19 |
| Actual Churn | 22 | 160 |

The group-aware split prevents identical feature profiles
from appearing in both the training and test partitions.

The test results are development-stage estimates.
They are not independent validation on new,
real-world customer data.

## Important Churn Predictors

Permutation importance identified the following important
features:

1. Tenure
2. Customer complaints
3. Number of addresses
4. Cashback amount
5. Satisfaction score
6. Preferred order category

Shorter tenure and customer complaints were associated
with higher churn risk.

These relationships are predictive associations,
not proof of causation.

## Streamlit Application Features

### Individual Customer Prediction

Enter 18 customer features and generate a churn score
using the trained Random Forest pipeline.

### Customer Analytics Dashboard

Explore historical churn patterns, customer counts,
satisfaction scores, complaints, and product categories.

### Model Comparison

Compare machine-learning algorithms using accuracy,
precision, recall, F1-score, ROC-AUC, and PR-AUC.

### Customer Insights

View permutation feature importance and a SHAP
summary visualization.

### Bulk Customer Prediction

Upload a CSV containing the required customer features.

The system generates:
- Churn scores
- Binary churn predictions
- Risk categories
- Downloadable prediction results

Model scores are not guaranteed to be calibrated
real-world churn probabilities.

## Installation

Use Python with the dependencies listed in
requirements.txt.

From the project directory, create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The saved model requires scikit-learn 1.6.1
for compatibility with its serialized pipeline.

## Running the Application

From the project directory, execute:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the application at:

http://localhost:8501

## Required Project Files

- app.py
- churn_prediction_model.joblib
- ecommerce_churn_cleaned (8).csv
- requirements.txt
- pages/1_Dashboard.py
- pages/2_Model_Comparison.py
- pages/3_Customer_Insights.py
- pages/4_Bulk_Prediction.py
- assets/shap_summary.png
- .streamlit/config.toml

## Limitations

- Model performance is based on historical customer data.
- Some identical customer feature profiles exist in the dataset.
- Predictions may be unreliable for unfamiliar customer profiles.
- The current model scores have not been probability-calibrated.
- Predictive feature associations should not be interpreted
  as causal relationships.
- Real-world deployment would require additional validation,
  monitoring, and model maintenance.

## Conclusion

The project demonstrates how machine-learning methods
can identify patterns associated with e-commerce customer
churn.

The Random Forest model, explainability analysis, and
Streamlit application provide an end-to-end workflow
for customer churn analysis and prediction.

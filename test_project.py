
from pathlib import Path
import pandas as pd
import joblib

project_dir = Path(__file__).resolve().parent


required_files = [
    "app.py",
    "churn_prediction_model.joblib",
    "requirements.txt",
    "README.md",
    "pages/Dashboard.py",
    "pages/Model_Comparison.py",
    "pages/Customer_Insights.py",
    "pages/Bulk_Prediction.py",
    "assets/shap_summary.png"
]


for file in required_files:
    assert (project_dir / file).exists(), f"Missing file: {file}"

print("All required project files found.")

package = joblib.load(
    project_dir / "churn_prediction_model.joblib"
)

model = package["pipeline"]
features = package["feature_columns"]
threshold = package["threshold"]

print("Trained Random Forest loaded successfully.")
print("Input features:", len(features))
print("Classification threshold:", threshold)

csv_files = list(
    project_dir.glob("ecommerce_churn_cleaned*.csv")
)

assert csv_files, "Cleaned dataset not found."

df = pd.read_csv(csv_files[0])

X = df[features].head(10).copy()

for col, mapping in package["category_mappings"].items():
    X[col] = X[col].replace(mapping)

scores = model.predict_proba(X)[:, 1]
predictions = (scores >= threshold).astype(int)

assert len(scores) == 10
assert all(0 <= score <= 1 for score in scores)

print("Dataset loaded:", len(df), "customers")
print("Sample predictions generated:", len(predictions))
print("Predicted churn cases:", int(predictions.sum()))
print("ALL CHECKS PASSED!")

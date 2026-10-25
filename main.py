from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
regression_model = joblib.load(ROOT / "models" / "regression.joblib")
classification_model = joblib.load(ROOT / "models" / "classification.joblib")
decision_tree_model = joblib.load(ROOT / "models" / "decision_tree.joblib")
app = FastAPI(title="Telco Customer ML Demo")

class RegressionInput(BaseModel):
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0, le=1000)

class ClassificationInput(BaseModel):
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(ge=0, le=1000)
    contract: str
    internet_service: str
    payment_method: str
    paperless_billing: str
    tech_support: str

@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    row = pd.DataFrame([{
        "tenure": data.tenure,
        "MonthlyCharges": data.monthly_charges,
    }])
    result = regression_model.predict(row)[0]
    chart_rows = pd.DataFrame({
        "tenure": list(range(0, 73, 12)),
        "MonthlyCharges": [data.monthly_charges] * 7,
    })
    trend = regression_model.predict(chart_rows)
    return {
        "prediction": round(float(result), 2),
        "target": "TotalCharges",
        "trend": [
            {"tenure": month, "prediction": round(float(value), 2)}
            for month, value in zip(chart_rows["tenure"], trend)
        ],
    }

@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = {
        "tenure": data.tenure,
        "MonthlyCharges": data.monthly_charges,
        "Contract": data.contract,
        "InternetService": data.internet_service,
        "PaymentMethod": data.payment_method,
        "PaperlessBilling": data.paperless_billing,
        "TechSupport": data.tech_support,
    }
    customer = pd.DataFrame([row])
    prediction = int(classification_model.predict(customer)[0])
    probability = float(classification_model.predict_proba(customer)[0][1])
    tree_preprocessor = decision_tree_model.named_steps["preprocess"]
    tree = decision_tree_model.named_steps["model"]
    transformed = tree_preprocessor.transform(customer)
    tree_prediction = int(tree.predict(transformed)[0])
    classes = list(tree.classes_)
    tree_probability = float(tree.predict_proba(transformed)[0][classes.index(1)])

    feature_names = tree_preprocessor.get_feature_names_out()
    numeric_scaler = tree_preprocessor.named_transformers_["numeric"]
    numeric_columns = ["tenure", "MonthlyCharges"]
    tree_data = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
    path = []
    node = 0
    while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
        feature_index = tree.tree_.feature[node]
        value = float(tree_data[feature_index])
        threshold = float(tree.tree_.threshold[node])
        raw_name = str(feature_names[feature_index])
        if raw_name.startswith("numeric__"):
            numeric_name = raw_name.removeprefix("numeric__")
            numeric_index = numeric_columns.index(numeric_name)
            threshold = threshold * float(numeric_scaler.scale_[numeric_index]) + float(numeric_scaler.mean_[numeric_index])
            feature_name = "Tenure" if numeric_name == "tenure" else "Monthly charges ($)"
        else:
            category = raw_name.removeprefix("categorical__")
            column, _, category_value = category.partition("_")
            labels = {"Contract": "Contract", "InternetService": "Internet service", "PaymentMethod": "Payment method", "PaperlessBilling": "Paperless billing", "TechSupport": "Tech support"}
            feature_name = f"{labels.get(column, column)} = {category_value}"
        went_left = value <= threshold
        path.append({
            "rule": f"{feature_name} ≤ {threshold:.2f}",
            "branch": "yes" if went_left else "no",
        })
        node = tree.tree_.children_left[node] if went_left else tree.tree_.children_right[node]

    return {
        "prediction": "Yes" if prediction else "No",
        "churn_probability": round(probability, 4),
        "tree_prediction": "Yes" if tree_prediction else "No",
        "tree_churn_probability": round(tree_probability, 4),
        "tree_path": path,
    }

app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

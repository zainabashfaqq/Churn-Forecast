from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "telco_churn.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
df = pd.read_csv(DATA)

# Regression: estimate total charges from tenure and monthly charges.
reg = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
reg["TotalCharges"] = pd.to_numeric(reg["TotalCharges"], errors="coerce")
reg = reg.dropna()
Xr, yr = reg[["tenure", "MonthlyCharges"]], reg["TotalCharges"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
regression = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
regression.fit(Xr_train, yr_train)
r2 = r2_score(yr_test, regression.predict(Xr_test))
joblib.dump(regression, MODELS / "regression.joblib")

# Classification: predict churn using a compact set of customer attributes.
features = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "TechSupport"]
Xc, yc = df[features], df["Churn"].map({"No": 0, "Yes": 1})
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), ["tenure", "MonthlyCharges"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "TechSupport"]),
    ])

classification = Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
classification.fit(Xc_train, yc_train)
accuracy = accuracy_score(yc_test, classification.predict(Xc_test))
joblib.dump(classification, MODELS / "classification.joblib")
decision_tree = Pipeline([
    ("preprocess", make_preprocessor()),
    ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, class_weight="balanced", random_state=42)),
])
decision_tree.fit(Xc_train, yc_train)
tree_accuracy = accuracy_score(yc_test, decision_tree.predict(Xc_test))
joblib.dump(decision_tree, MODELS / "decision_tree.joblib")
print(f"Regression R² (holdout): {r2:.4f}")
print(f"Classification accuracy (holdout): {accuracy:.4f}")
print(f"Decision tree accuracy (holdout): {tree_accuracy:.4f}")

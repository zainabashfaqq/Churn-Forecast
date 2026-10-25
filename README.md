# Telco customer ML demo

Train the regression, logistic regression, and decision tree models and print holdout metrics:

```powershell
python -m pip install -r requirements.txt
python train_models.py
```

Run the API and page:

```powershell
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000. The API accepts JSON at `/predict/regression` and `/predict/classification`; interactive schema docs are at `/docs`.

The regression model estimates `TotalCharges` from `tenure` and `MonthlyCharges`. The churn endpoint compares logistic regression with a decision tree using tenure, monthly charges, contract, internet service, payment method, paperless billing, and tech support. The frontend draws a charge forecast, compares churn probabilities, and explains the decision tree route for the submitted customer. Models use an 80/20 holdout split (random seed 42); metrics are printed by the training script. Fitted pipelines are saved under `models/` (`regression.joblib`, `classification.joblib`, and `decision_tree.joblib`).

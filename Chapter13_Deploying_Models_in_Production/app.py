from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API", version="1.0.0")
model = joblib.load("models/churn_pipeline.joblib")


class Customer(BaseModel):
    tenure_months: float = Field(ge=0)
    monthly_fee: Optional[float] = Field(default=None, ge=0)
    support_calls: Optional[float] = Field(default=None, ge=0)
    contract: str
    payment: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5}

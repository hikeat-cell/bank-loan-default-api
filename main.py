from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Create FastAPI application
app = FastAPI(
    title="Bank Loan Default Prediction API",
    description="Random Forest model for predicting probability of loan default",
    version="1.0"
)


# Define the customer information expected from Excel/API
class CustomerData(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


# Home page
@app.get("/")
def home():
    return {
        "message": "Bank Loan Default Prediction API",
        "endpoint": "/predict"
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerData):

    # Create dataframe using the same variables
    # and order used to train the model
    input_data = pd.DataFrame([{
        "AGE": customer.AGE,
        "EMPLOY": customer.EMPLOY,
        "ADDRESS": customer.ADDRESS,
        "DEBTINC": customer.DEBTINC,
        "CREDDEBT": customer.CREDDEBT,
        "OTHDEBT": customer.OTHDEBT
    }])

    # Probability that DEFAULTER = 1
    probability = model.predict_proba(input_data)[0][1]

    return {
        "probability_of_default": round(float(probability), 4),
        "probability_percent": round(float(probability) * 100, 2)
    }
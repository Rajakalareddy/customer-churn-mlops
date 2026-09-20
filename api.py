from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd
from pydantic import BaseModel

# Create the FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained machine learning model.",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "churn_model.joblib"

model = joblib.load(MODEL_PATH)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "model": "Logistic Regression"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
# Customer input data for churn prediction
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # Customer churn prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert customer information into a DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Generate prediction and churn probability
    prediction = int(model.predict(customer_df)[0])

    churn_probability = float(
        model.predict_proba(customer_df)[0][1]
    )

    # Return prediction results
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percentage": round(
            churn_probability * 100, 2
        )
    }
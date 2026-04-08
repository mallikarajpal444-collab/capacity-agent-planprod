# ================================
# FASTAPI FOR ML MODEL
# ================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ================================
# LOAD MODEL
# ================================

model = joblib.load(os.path.join(os.getcwd(), "alert_model.joblib"))
label_encoder = joblib.load(os.path.join(os.getcwd(), "label_encoder.joblib"))

# ================================
# INIT APP
# ================================

app = FastAPI(title="Production Risk API")

# ================================
# INPUT SCHEMA
# ================================

class InputData(BaseModel):
    utilization_pct: float
    delay_hrs: float
    throughput_deviation_pct: float
    efficiency: float
    delay_ratio: float
    shortage_probability: float
    supplier_otif: float

# ================================
# HEALTH CHECK
# ================================

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# ================================
# PREDICTION ENDPOINT
# ================================

@app.post("/predict")
def predict(data: InputData):
    
    # Convert input to model format
    input_array = np.array([[
        data.utilization_pct,
        data.delay_hrs,
        data.throughput_deviation_pct,
        data.efficiency,
        data.delay_ratio,
        data.shortage_probability,
        data.supplier_otif
    ]])

    # Predict
    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)

    # Decode label
    label = label_encoder.inverse_transform(prediction)[0]

    # Confidence score (max probability)
    confidence = float(np.max(probabilities))

    # Format probabilities nicely
    prob_dict = {
        label_encoder.classes_[i]: float(probabilities[0][i])
        for i in range(len(label_encoder.classes_))
    }

    return {
        "alert_level": label,
        "confidence": round(confidence, 3),
        "probabilities": prob_dict
    }
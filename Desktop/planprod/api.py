# ================================
# FASTAPI FOR ML MODEL (FINAL)
# ================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ================================
# LOAD MODEL
# ================================

model = joblib.load(os.path.join(os.getcwd(), "alert_model.joblib"))
label_encoder = joblib.load(os.path.join(os.getcwd(), "label_encoder.joblib"))

# ================================
# INIT APP
# ================================

app = FastAPI(title="Capacity & Schedule API")

# ================================
# SCHEDULING FUNCTION
# ================================

def compute_schedule(processing_time, queue_length, due_in_hrs):
    total_time = processing_time * (queue_length + 1)
    delay = total_time - due_in_hrs

    if delay > 2:
        risk = "HIGH"
    elif delay > 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return total_time, delay, risk

# ================================
# INPUT SCHEMA
# ================================

class InputData(BaseModel):
    # ML inputs
    utilization_pct: float
    delay_hrs: float
    throughput_deviation_pct: float
    efficiency: float
    delay_ratio: float
    shortage_probability: float
    supplier_otif: float

    # Scheduling inputs
    processing_time_hrs: float
    queue_length: int
    due_in_hrs: float

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

    # ----------------------------
    # ML INPUT
    # ----------------------------
    input_array = np.array([[
        data.utilization_pct,
        data.delay_hrs,
        data.throughput_deviation_pct,
        data.efficiency,
        data.delay_ratio,
        data.shortage_probability,
        data.supplier_otif
    ]])

    # ----------------------------
    # ML PREDICTION
    # ----------------------------
    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)

    label = label_encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities))

    # ----------------------------
    # SCHEDULING LOGIC
    # ----------------------------
    total_time, delay, schedule_risk = compute_schedule(
        data.processing_time_hrs,
        data.queue_length,
        data.due_in_hrs
    )

    # ----------------------------
    # FORMAT PROBABILITIES
    # ----------------------------
    prob_dict = {
        label_encoder.classes_[i]: float(probabilities[0][i])
        for i in range(len(label_encoder.classes_))
    }

    # ----------------------------
    # RESPONSE
    # ----------------------------
    return {
        "alert_level": label,
        "confidence": round(confidence, 3),
        "probabilities": prob_dict,

        "expected_completion_hrs": total_time,
        "expected_delay_hrs": max(0, delay),
        "schedule_risk": schedule_risk
    }
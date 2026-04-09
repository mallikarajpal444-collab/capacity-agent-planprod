# ================================
# CAPACITY + SCHEDULE + DEMAND INTEGRATED API
# ================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import joblib
import numpy as np
import os

# ================================
# LOAD MODELS
# ================================

clf = joblib.load(os.path.join(os.getcwd(), "capacity_classifier.joblib"))
reg = joblib.load(os.path.join(os.getcwd(), "delay_regressor.joblib"))
label_encoder = joblib.load(os.path.join(os.getcwd(), "label_encoder.joblib"))

# ================================
# INIT APP
# ================================

app = FastAPI(title="Capacity & Schedule Control Tower API")

# ================================
# HELPER FUNCTIONS
# ================================

def compute_capacity_left(utilization):
    return max(0, 100 - utilization)


def compute_schedule(predicted_delay, processing_time, queue_length, due_in_hrs):
    total_time = processing_time * (queue_length + 1) + predicted_delay
    delay = total_time - due_in_hrs

    if delay > 2:
        risk = "HIGH"
    elif delay > 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return total_time, delay, risk


def suggest_best_workcentre(work_centres):
    if not work_centres:
        return None

    best = min(work_centres, key=lambda wc: wc["utilization"])

    return {
        "id": best["id"],
        "utilization": best["utilization"],
        "capacity_left": 100 - best["utilization"]
    }

# ================================
# INPUT SCHEMA
# ================================

class InputData(BaseModel):

    # 🔥 Demand Agent Inputs
    forecast_qty: float
    shortage_probability: float

    # 🔥 Work centre state
    utilization_pct: float
    throughput_deviation_pct: float
    efficiency: float
    supplier_otif: float

    # 🔥 Scheduling
    queue_length: int
    due_in_hrs: float

    # 🔥 Supabase Work Centres
    work_centres: List[Dict]

# ================================
# HEALTH CHECK
# ================================

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# ================================
# PREDICT ENDPOINT
# ================================

@app.post("/predict")
def predict(data: InputData):

    # ----------------------------
    # DEMAND → PROCESSING LOAD
    # ----------------------------
    processing_time_hrs = data.forecast_qty * 0.02

    # ----------------------------
    # FEATURE VECTOR (ML INPUT)
    # ----------------------------
    features = np.array([[
        data.utilization_pct,
        data.throughput_deviation_pct,
        data.efficiency,
        data.shortage_probability,
        data.supplier_otif
    ]])

    # ----------------------------
    # MODEL PREDICTIONS
    # ----------------------------
    pred_class = clf.predict(features)
    pred_delay = reg.predict(features)

    alert_level = label_encoder.inverse_transform(pred_class)[0]
    predicted_delay = float(pred_delay[0])

    # ----------------------------
    # CAPACITY
    # ----------------------------
    capacity_left = compute_capacity_left(data.utilization_pct)

    # ----------------------------
    # SCHEDULING
    # ----------------------------
    total_time, delay, schedule_risk = compute_schedule(
        predicted_delay,
        processing_time_hrs,
        data.queue_length,
        data.due_in_hrs
    )

    # ----------------------------
    # BEST WORK CENTRE
    # ----------------------------
    best_wc = suggest_best_workcentre(data.work_centres)

    # ----------------------------
    # DECISION LOGIC
    # ----------------------------
    if best_wc and (schedule_risk == "HIGH" or alert_level == "RED"):
        action = "SHIFT"
        reason = f"Shift to {best_wc['id']} (utilization {best_wc['utilization']}%)"
    elif alert_level == "RED":
        action = "OPTIMIZE_LOAD"
        reason = "Reduce load on current work centre"
    else:
        action = "MONITOR"
        reason = "System stable"

    # ----------------------------
    # FINAL STATUS
    # ----------------------------
    status = "CRITICAL" if alert_level == "RED" or schedule_risk == "HIGH" else "NORMAL"

    # ----------------------------
    # RESPONSE
    # ----------------------------
    return {
        "status": status,

        "demand": {
            "forecast_qty": data.forecast_qty,
            "processing_time_hrs": round(processing_time_hrs, 2)
        },

        "capacity": {
            "current_utilization": data.utilization_pct,
            "capacity_left": capacity_left
        },

        "prediction": {
            "alert_level": alert_level,
            "predicted_delay_hrs": round(predicted_delay, 2)
        },

        "schedule": {
            "expected_completion_hrs": round(total_time, 2),
            "expected_delay_hrs": round(max(0, delay), 2),
            "risk": schedule_risk
        },

        "recommendation": {
            "action": action,
            "target_work_centre": best_wc["id"] if best_wc else None,
            "target_utilization": best_wc["utilization"] if best_wc else None,
            "target_capacity_left": best_wc["capacity_left"] if best_wc else None,
            "reason": reason
        }
    }
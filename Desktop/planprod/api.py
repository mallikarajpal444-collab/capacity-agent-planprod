# ================================
# CAPACITY + SCHEDULE + DEMAND INTEGRATED API
# ================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import os

# ================================
# LOAD MODELS
# ================================

clf           = joblib.load(os.path.join(os.getcwd(), "capacity_classifier.joblib"))
reg           = joblib.load(os.path.join(os.getcwd(), "delay_regressor.joblib"))
label_encoder = joblib.load(os.path.join(os.getcwd(), "label_encoder.joblib"))

# ================================
# INIT APP
# ================================

app = FastAPI(title="Capacity & Schedule Control Tower API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# SCHEMAS
# ================================

class WorkCentre(BaseModel):
    id: str
    utilization_pct: float
    efficiency: Optional[float] = 1.0
    queue_length: Optional[int] = 0

class InputData(BaseModel):
    # From Demand Agent
    forecast_qty: float
    shortage_probability: float
    demand_gap: Optional[float] = None          # optional — pass if demand agent sends it

    # From Supabase (primary / current work centre being evaluated)
    utilization_pct: float
    throughput_deviation_pct: float
    efficiency: float
    supplier_otif: float
    queue_length: int
    due_in_hrs: float

    # All available work centres from Supabase
    work_centres: List[WorkCentre]

# ================================
# HELPER FUNCTIONS
# ================================

def compute_capacity_left(utilization: float) -> float:
    return round(max(0.0, 100.0 - utilization), 2)


def compute_schedule(predicted_delay: float, processing_time: float,
                     queue_length: int, due_in_hrs: float):
    total_time = processing_time * (queue_length + 1) + predicted_delay
    delay      = total_time - due_in_hrs

    if delay > 2:
        risk = "HIGH"
    elif delay > 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return round(total_time, 2), round(max(0.0, delay), 2), risk


def suggest_best_workcentre(work_centres: List[WorkCentre]):
    if not work_centres:
        return None
    best = min(work_centres, key=lambda wc: wc.utilization_pct)
    return {
        "id":             best.id,
        "utilization_pct": best.utilization_pct,
        "capacity_left":  round(100 - best.utilization_pct, 2),
    }

# ================================
# HEALTH CHECK
# ================================

@app.get("/")
def home():
    return {"status": "ok", "service": "Capacity & Schedule Control Tower API"}

# ================================
# PREDICT ENDPOINT
# ================================

@app.post("/predict")
def predict(data: InputData):

    # ── DEMAND → LOAD ──────────────────────────────────────────
    processing_time_hrs = round(data.forecast_qty * 0.02, 2)

    # ── FEATURE VECTOR ─────────────────────────────────────────
    features = np.array([[
        data.utilization_pct,
        data.throughput_deviation_pct,
        data.efficiency,
        data.shortage_probability,
        data.supplier_otif,
    ]])

    # ── ML INFERENCE ───────────────────────────────────────────
    alert_level     = label_encoder.inverse_transform(clf.predict(features))[0]
    predicted_delay = round(float(reg.predict(features)[0]), 2)

    # ── CAPACITY ───────────────────────────────────────────────
    capacity_left = compute_capacity_left(data.utilization_pct)

    # ── SCHEDULE ───────────────────────────────────────────────
    completion_hrs, expected_delay_hrs, schedule_risk = compute_schedule(
        predicted_delay, processing_time_hrs, data.queue_length, data.due_in_hrs
    )

    # ── WORK CENTRE SELECTION ──────────────────────────────────
    best_wc = suggest_best_workcentre(data.work_centres)

    # ── DECISION ENGINE ────────────────────────────────────────
    is_critical = alert_level == "RED" or schedule_risk == "HIGH"

    if best_wc and is_critical:
        action = "SHIFT"
        reason = f"High risk detected — shift to {best_wc['id']} (utilization {best_wc['utilization_pct']}%)"
    elif alert_level == "RED":
        action = "OPTIMIZE_LOAD"
        reason = "Capacity overloaded — reduce load on current work centre"
    elif data.demand_gap is not None and data.demand_gap > 0:
        action = "EXPEDITE"
        reason = f"Demand gap of {data.demand_gap} units — expedite production"
    else:
        action = "MONITOR"
        reason = "System stable — continue monitoring"

    status = "CRITICAL" if is_critical else "NORMAL"

    # ── RESPONSE ───────────────────────────────────────────────
    # Flat structure — easy to extract in n8n Set node
    return {
        # Top-level summary
        "status":                   status,
        "action":                   action,
        "reason":                   reason,

        # Demand
        "forecast_qty":             data.forecast_qty,
        "demand_gap":               data.demand_gap,
        "processing_time_hrs":      processing_time_hrs,

        # Capacity
        "current_utilization_pct":  data.utilization_pct,
        "capacity_left_pct":        capacity_left,

        # ML Predictions
        "alert_level":              alert_level,
        "predicted_delay_hrs":      predicted_delay,

        # Schedule
        "expected_completion_hrs":  completion_hrs,
        "expected_delay_hrs":       expected_delay_hrs,
        "schedule_risk":            schedule_risk,

        # Work Centre Recommendation
        "recommended_wc_id":        best_wc["id"] if best_wc else None,
        "recommended_wc_utilization": best_wc["utilization_pct"] if best_wc else None,
        "recommended_wc_capacity_left": best_wc["capacity_left"] if best_wc else None,
    }

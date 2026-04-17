# ================================
# CAPACITY + SCHEDULE + DEMAND INTEGRATED API (WITH ORDER TRACKING)
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
    utilization_pct: Optional[float] = None
    utilization: Optional[float] = None

    def get_utilization(self) -> float:
        return self.utilization_pct if self.utilization_pct is not None else (self.utilization or 0.0)


class InputData(BaseModel):
    # ✅ NEW: Production Order Tracking
    production_order_no: str

    # Demand Agent
    forecast_qty: float
    shortage_probability: float
    demand_gap: Optional[float] = None

    # OPTIONAL
    current_work_centre: Optional[str] = None

    # Current WC data
    utilization_pct: float
    throughput_deviation_pct: float
    efficiency: float
    supplier_otif: float
    queue_length: int
    due_in_hrs: float

    # All work centres
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


def suggest_best_workcentre(work_centres: List[WorkCentre], current_wc_id: Optional[str]):

    if not work_centres:
        return None

    filtered = [wc for wc in work_centres if wc.id != current_wc_id] if current_wc_id else work_centres

    if not filtered:
        return None

    best = min(filtered, key=lambda wc: wc.get_utilization())
    util = best.get_utilization()

    return {
        "id": best.id,
        "utilization_pct": util,
        "capacity_left": round(100 - util, 2),
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

    # ----------------------------
    # CURRENT WC FALLBACK
    # ----------------------------
    current_wc = data.current_work_centre or (
        data.work_centres[0].id if data.work_centres else None
    )

    # ----------------------------
    # DEMAND → PROCESSING TIME
    # ----------------------------
    processing_time_hrs = round(data.forecast_qty * 0.02, 2)

    # ----------------------------
    # FEATURE VECTOR
    # ----------------------------
    supplier_otif = data.supplier_otif / 100 if data.supplier_otif > 1 else data.supplier_otif

    features = np.array([[  
        data.utilization_pct,
        data.throughput_deviation_pct,
        data.efficiency,
        data.shortage_probability,
        supplier_otif,
    ]])

    # ----------------------------
    # ML PREDICTION
    # ----------------------------
    alert_level     = label_encoder.inverse_transform(clf.predict(features))[0]
    predicted_delay = round(float(reg.predict(features)[0]), 2)

    # ----------------------------
    # CAPACITY
    # ----------------------------
    capacity_left = compute_capacity_left(data.utilization_pct)

    # ----------------------------
    # SCHEDULE
    # ----------------------------
    completion_hrs, expected_delay_hrs, schedule_risk = compute_schedule(
        predicted_delay,
        processing_time_hrs,
        data.queue_length,
        data.due_in_hrs
    )

    # ----------------------------
    # WORK CENTRE SELECTION
    # ----------------------------
    best_wc = suggest_best_workcentre(data.work_centres, current_wc)

    # ----------------------------
    # DECISION ENGINE
    # ----------------------------
    is_critical = alert_level == "RED" or schedule_risk == "HIGH"

    if best_wc and is_critical:
        action = "SHIFT"
        reason = f"Shift to {best_wc['id']} (utilization {best_wc['utilization_pct']}%)"

    elif is_critical:
        action = "OPTIMIZE_LOAD"
        reason = "High risk but no better work centre available"

    elif data.demand_gap and data.demand_gap > 0:
        action = "EXPEDITE"
        reason = f"Demand gap of {data.demand_gap} units"

    else:
        action = "MONITOR"
        reason = "System stable"

    status = "CRITICAL" if is_critical else "NORMAL"

    # ----------------------------
    # RESPONSE
    # ----------------------------
    return {
        "production_order_no": data.production_order_no,  # ✅ INCLUDED

        "status": status,
        "action": action,
        "reason": reason,

        "demand": {
            "forecast_qty": data.forecast_qty,
            "processing_time_hrs": processing_time_hrs
        },

        "capacity": {
            "current_work_centre": current_wc,
            "current_utilization": data.utilization_pct,
            "capacity_left": capacity_left
        },

        "prediction": {
            "alert_level": alert_level,
            "predicted_delay_hrs": predicted_delay
        },

        "schedule": {
            "expected_completion_hrs": completion_hrs,
            "expected_delay_hrs": expected_delay_hrs,
            "risk": schedule_risk
        },

        "recommendation": {
            "target_work_centre": best_wc["id"] if best_wc else None,
            "target_utilization": best_wc["utilization_pct"] if best_wc else None,
            "target_capacity_left": best_wc["capacity_left"] if best_wc else None,
            "reason": reason
        }
    }
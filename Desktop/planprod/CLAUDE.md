# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run locally (development):**
```bash
uvicorn api:app --reload
```

**Run for production (as defined in Procfile):**
```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/

# Prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_qty": 100, "shortage_probability": 0.2, "utilization_pct": 75, "throughput_deviation_pct": 5, "efficiency": 0.9, "supplier_otif": 0.85, "queue_length": 10, "due_in_hrs": 24, "work_centres": [{"name": "WC1", "utilization_pct": 70}]}'
```

## Architecture

This is a FastAPI-based ML inference API serving as a "Control Tower" for manufacturing capacity and scheduling decisions.

### Core Flow

```
POST /predict
  → InputData schema validation
  → Feature vector (5 features: utilization, throughput_deviation, efficiency, shortage_probability, supplier_otif)
  → capacity_classifier.joblib  → alert level (RED/YELLOW/GREEN via label_encoder.joblib)
  → delay_regressor.joblib      → predicted delay in hours
  → Decision logic              → action (SHIFT / OPTIMIZE_LOAD / MONITOR)
  → JSON response
```

### ML Models

Three pre-trained joblib models are committed to the repo and loaded at startup:
- `capacity_classifier.joblib` — classifies alert level
- `delay_regressor.joblib` — predicts delay in hours
- `label_encoder.joblib` — decodes classifier output to string labels

These are loaded once at module level in [api.py](api.py) and reused across requests.

### Key Logic (`/predict` endpoint)

- Processing time is derived as `forecast_qty × 0.02` hours
- Schedule risk (LOW/MEDIUM/HIGH) is computed from predicted delay vs. due time with queue buffer
- Action is determined: `SHIFT` if a better work centre exists, `OPTIMIZE_LOAD` if utilization > 80%, else `MONITOR`
- Overall status is `CRITICAL` if alert level is `RED`, otherwise `NORMAL`

### Work Centre Data

Work centres are passed per-request as a list in the `InputData` payload (sourced from Supabase by the caller). The API selects the work centre with the lowest utilization as the recommended target.

### Deployment

Heroku-compatible via `Procfile`. The `$PORT` environment variable is set by the platform.

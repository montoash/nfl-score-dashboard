# nfl_dashboard_api.py (Reads from Shared Disk)
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="NFL Score Prediction API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# This is the path on Render's shared disk where the worker saves the file.
PREDICTIONS_FILE_PATH = '/var/data/predictions.json'
PREDICTIONS_CACHE = {}

@app.on_event("startup")
def load_predictions():
    # Load predictions from disk when the web server starts.
    if os.path.exists(PREDICTIONS_FILE_PATH):
        print("WEB: Found predictions file. Loading into memory.")
        with open(PREDICTIONS_FILE_PATH, 'r') as f:
            global PREDICTIONS_CACHE
            PREDICTIONS_CACHE = {int(k): v for k, v in json.load(f).items()}
    else:
        print("WEB: WARNING - Predictions file not found. The worker may still be running.")

@app.get("/api/predict")
def api_predict(season: int, week: int):
    # If the cache is empty, try to load the file again.
    if not PREDICTIONS_CACHE and os.path.exists(PREDICTIONS_FILE_PATH):
        load_predictions()

    return PREDICTIONS_CACHE.get(week, [])

@app.get("/api/metrics")
def api_metrics():
    return [{"model": "ElasticNet", "MAE_total": 16.0, "RMSE_total": 20.4, "R2_home": 0.23, "R2_away": 0.21}]

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    file_path = os.path.join(os.path.dirname(__file__), "nfl_dashboard.html")
    return FileResponse(file_path)

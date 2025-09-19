#
# MODIFIED NFL DASHBOARD API
# This version pre-calculates all predictions on startup to avoid timeouts.
#

from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np

# --- 1. Import the modelling pipeline ---
# We keep the try/except block just in case, but the logs show it's working.
try:
    print("API is starting, attempting to import modelling libraries...")
    from nfl_score_prediction import (
        load_game_data,
        compute_team_statistics,
        prepare_model_data,
        evaluate_models,
        train_final_model,
    )
    import nfl_data_py as nfl
    HAS_NFL_DATA = True
    print("--> SUCCESS: All modelling libraries imported correctly.")
except Exception as e:
    HAS_NFL_DATA = False
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! CRITICAL ERROR: FAILED to import modelling libraries.     !!!")
    print(f"!!! Error details: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


app = FastAPI(title="NFL Score Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Create global variables to store our pre-calculated results ---
# We will fill these variables when the server starts.
PREDICTIONS_CACHE: Dict[int, List[Dict[str, Any]]] = {}
METRICS_CACHE: List[Dict[str, Any]] = []


# --- 3. Create a "startup" event to run the model ONCE ---
@app.on_event("startup")
def run_model_on_startup():
    """
    This function runs only once when the server starts.
    It calculates predictions for all weeks and stores them in the cache.
    """
    print("Server is starting up. Running model to pre-calculate all predictions...")

    # If the libraries didn't import, we can't run the model.
    if not HAS_NFL_DATA:
        print("Cannot run model because libraries failed to import. API will not have data.")
        return

    try:
        # --- Pre-calculate Metrics ---
        global METRICS_CACHE
        print("Calculating model performance metrics...")
        games_full_metrics = load_game_data(2015, 2024, include_unplayed=False)
        games_with_stats_metrics = compute_team_statistics(games_full_metrics)
        X_metrics, Y_metrics = prepare_model_data(games_with_stats_metrics)
        if len(X_metrics) > 0:
            metrics_df = evaluate_models(X_metrics, Y_metrics)
            METRICS_CACHE = metrics_df.to_dict(orient='records')
            print(f"--> SUCCESS: Calculated metrics for {len(METRICS_CACHE)} models.")
        else:
            print("--> WARNING: No data available to calculate metrics.")

        # --- Pre-calculate Weekly Predictions for the 2025 Season ---
        global PREDICTIONS_CACHE
        prediction_season = 2025 # The season we want to predict
        print(f"Loading data for the {prediction_season} prediction season...")
        games_full = load_game_data(2015, prediction_season, include_unplayed=True)
        games_with_stats = compute_team_statistics(games_full)
        X_full, Y_full = prepare_model_data(games_with_stats)

        # Loop through every week (1-18) and generate predictions
        for week in range(1, 19):
            print(f"  - Generating predictions for Week {week}...")
            train_mask = (
                games_with_stats['home_score'].notna() &
                ((games_with_stats['season'] < prediction_season) | ((games_with_stats['season'] == prediction_season) & (games_with_stats['week'] < week)))
            )
            pred_mask = (games_with_stats['season'] == prediction_season) & (games_with_stats['week'] == week)

            if pred_mask.sum() == 0:
                print(f"    - No games scheduled for Week {week}, skipping.")
                PREDICTIONS_CACHE[week] = []
                continue

            X_train = X_full.loc[train_mask].reset_index(drop=True)
            Y_train = Y_full.loc[train_mask].reset_index(drop=True)
            X_pred = X_full.loc[pred_mask]

            if len(X_train) > 0:
                means = X_train.mean()
                X_train = X_train.fillna(means)
                X_pred = X_pred.fillna(means)
            else: # For week 1, there's no prior training data in the same season
                 X_pred = X_pred.fillna(0)


            # In a real app, you would train one model here and reuse it.
            # For simplicity, we continue your pattern of re-evaluating.
            best_model_name = 'ElasticNet' # Defaulting to a fast model
            if len(X_train) > 0:
                metrics = evaluate_models(X_train, Y_train)
                best_model_name = metrics.sort_values('RMSE_total').iloc[0]['model']
            
            model = train_final_model(X_train, Y_train, model_name=best_model_name)
            preds = model.predict(X_pred)

            upcoming_games = games_with_stats.loc[pred_mask, ['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']].copy()
            upcoming_games['home_pred'] = preds[:, 0]
            upcoming_games['away_pred'] = preds[:, 1]
            upcoming_games['pred_spread'] = upcoming_games['home_pred'] - upcoming_games['away_pred']
            upcoming_games['pred_total'] = upcoming_games['home_pred'] + upcoming_games['away_pred']
            
            PREDICTIONS_CACHE[week] = upcoming_games.to_dict(orient='records')
        print("--> SUCCESS: All weekly predictions have been pre-calculated and cached.")

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL ERROR during startup model run.                !!!")
        print(f"!!! Error details: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- 4. Modify API endpoints to be FAST ---
# These endpoints now just read from the cache, which is instant.

@app.get("/api/predict")
def api_predict(season: int, week: int):
    """API endpoint to get pre-calculated predictions for a given week."""
    print(f"Received request for predictions: Season {season}, Week {week}")
    if week in PREDICTIONS_CACHE:
        return PREDICTIONS_CACHE[week]
    else:
        # This case handles if the week is out of range (e.g., > 18)
        return []

@app.get("/api/metrics")
def api_metrics():
    """API endpoint to retrieve pre-calculated model performance metrics."""
    print("Received request for metrics.")
    return METRICS_CACHE

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive dashboard HTML file."""
    file_path = os.path.join(os.path.dirname(__file__), "nfl_dashboard.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dashboard HTML not found")
    return FileResponse(file_path, media_type="text/html")


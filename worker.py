# worker.py
import json
import pandas as pd
import time
import os
from nfl_score_prediction import (
    load_game_data,
    compute_team_statistics,
    prepare_model_data,
    evaluate_models,
    train_final_model
)

# Render's persistent disk is mounted at /var/data
# We will save our prediction file here.
OUTPUT_FILE = '/var/data/predictions.json'

def generate_and_save_predictions():
    """
    Runs the full modeling pipeline and saves predictions to the shared disk.
    """
    print("WORKER: Starting prediction generation process...")

    all_predictions = {}
    prediction_season = 2025

    print("WORKER: Loading data...")
    games_full = load_game_data(2015, prediction_season, include_unplayed=True)
    games_with_stats = compute_team_statistics(games_full)
    X_full, Y_full = prepare_model_data(games_with_stats)

    train_mask = games_with_stats['season'] < prediction_season
    X_train = X_full.loc[train_mask].reset_index(drop=True)
    Y_train = Y_full.loc[train_mask].reset_index(drop=True)

    print("WORKER: Training the main model once on all historical data...")
    final_model = train_final_model(X_train, Y_train, model_name='ElasticNet')
    print("WORKER: --> Model training complete.")

    print("WORKER: Generating predictions for all weeks...")
    for week in range(1, 19):
        pred_mask = (games_with_stats['season'] == prediction_season) & (games_with_stats['week'] == week)
        if pred_mask.sum() == 0:
            all_predictions[week] = []
            continue

        X_pred = X_full.loc[pred_mask].fillna(X_train.mean())
        preds = final_model.predict(X_pred)

        upcoming_games = games_with_stats.loc[pred_mask, ['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line']].copy()
        upcoming_games['home_pred'] = preds[:, 0]
        upcoming_games['away_pred'] = preds[:, 1]
        upcoming_games['pred_spread'] = upcoming_games['home_pred'] - upcoming_games['away_pred']
        upcoming_games['pred_total'] = upcoming_games['home_pred'] + upcoming_games['away_pred']

        all_predictions[week] = upcoming_games.to_dict(orient='records')
        print(f"WORKER:   - Predictions for Week {week} generated.")

    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_predictions, f, indent=4)

    print(f"WORKER: --> SUCCESS: All predictions have been saved to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    # This loop will run forever.
    while True:
        # Check if the predictions file already exists. If not, generate it.
        if not os.path.exists(OUTPUT_FILE):
            print("WORKER: Predictions file not found. Starting generation.")
            generate_and_save_predictions()
        else:
            print(f"WORKER: Predictions file found at {OUTPUT_FILE}. Sleeping for 24 hours.")

        # Sleep for a day before checking again.
        # You could make this more complex to re-generate weekly, but this is simple and effective.
        time.sleep(60 * 60 * 24) 

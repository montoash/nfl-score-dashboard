"""
NFL Game Score Prediction
=========================

This module builds and evaluates machine‑learning models to predict the final
scores of NFL games. It relies on the ``nfl_data_py`` package to fetch
historical schedules and outcomes, then computes rolling statistics and Elo
ratings for each team to use as predictive features. Multiple models are
benchmarked via time‑series cross‑validation, and the best one is trained
on all available historical data.

The ``main`` function demonstrates a typical workflow for generating
predictions for a specific season and week. When imported, the module
exposes helper functions used by the API layer defined in
``nfl_dashboard_api.py``.

Note: This file was automatically generated from the original ``nfl_score_prediction (1) (6).py``
script provided by the user. The core functionality has been preserved
without the extraneous filename characters, making it suitable for
deployment on hosting services that do not allow spaces in module names.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import PoissonRegressor
from sklearn.neural_network import MLPRegressor

# Optional gradient boosting libraries
try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # type: ignore
except ImportError:
    LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor  # type: ignore
except ImportError:
    CatBoostRegressor = None  # type: ignore

try:
    import statsmodels.api as sm  # type: ignore
except ImportError:
    sm = None  # type: ignore

from collections import defaultdict


def load_game_data(start_year: int, end_year: int, include_unplayed: bool = False) -> pd.DataFrame:
    """Load NFL schedule and results for a range of seasons.

    Parameters
    ----------
    start_year : int
        The first season to include (e.g., 2015).
    end_year : int
        The last season to include (e.g., 2025).  This function will load
        data for all seasons in the inclusive range.
    include_unplayed : bool, default False
        If False, drop games without final scores.  If True, retain all
        games and leave score columns as NaN for unplayed games.

    Returns
    -------
    pd.DataFrame
        A dataframe sorted by season and week.  If ``include_unplayed``
        is False, only completed games are returned.  If True, both
        completed and scheduled games are returned.
    """
    years = list(range(start_year, end_year + 1))
    schedules = nfl.import_schedules(years)
    id_cols = [
        'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
        'ftn', 'away_qb_id', 'home_qb_id', 'stadium_id'
    ]
    schedules = schedules.drop(columns=[c for c in id_cols if c in schedules.columns])
    if not include_unplayed:
        schedules = schedules.dropna(subset=['home_score', 'away_score', 'result'])
    schedules = schedules.sort_values(['season', 'week']).reset_index(drop=True)
    return schedules


def compute_team_statistics(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling offensive and defensive statistics for each team.

    For each team and for each game, this function calculates statistics
    based solely on prior games in the same season. Features include
    rolling averages over the last five games, season‑to‑date averages, and
    a win rate. Elo ratings are also merged into the returned dataframe.
    """
    df = games.copy()
    home_cols = {
        'team': df['home_team'],
        'opponent': df['away_team'],
        'points_for': df['home_score'],
        'points_against': df['away_score'],
        'is_home': 1,
    }
    away_cols = {
        'team': df['away_team'],
        'opponent': df['home_team'],
        'points_for': df['away_score'],
        'points_against': df['home_score'],
        'is_home': 0,
    }
    long_games = pd.concat([
        df[['season', 'week', 'game_id']].assign(**home_cols),
        df[['season', 'week', 'game_id']].assign(**away_cols)
    ], ignore_index=True)
    long_games = long_games.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    window = 5
    long_games['offense_avg_last5'] = (
        long_games.groupby(['team', 'season'])['points_for']
        .transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    long_games['defense_avg_last5'] = (
        long_games.groupby(['team', 'season'])['points_against']
        .transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    long_games['offense_avg_season'] = (
        long_games.groupby(['team', 'season'])['points_for']
        .transform(lambda x: x.shift().expanding(min_periods=1).mean())
    )
    long_games['defense_avg_season'] = (
        long_games.groupby(['team', 'season'])['points_against']
        .transform(lambda x: x.shift().expanding(min_periods=1).mean())
    )
    long_games['won'] = (long_games['points_for'] > long_games['points_against']).astype(int)
    long_games['win_rate_last5'] = (
        long_games.groupby(['team', 'season'])['won']
        .transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
    )
    stats_columns = [
        'offense_avg_last5', 'defense_avg_last5',
        'offense_avg_season', 'defense_avg_season', 'win_rate_last5'
    ]
    long_lookup = long_games[['game_id', 'team'] + stats_columns]
    df = df.merge(
        long_lookup.add_suffix('_home'),
        left_on=['game_id', 'home_team'], right_on=['game_id_home', 'team_home'],
        how='left'
    )
    df = df.merge(
        long_lookup.add_suffix('_away'),
        left_on=['game_id', 'away_team'], right_on=['game_id_away', 'team_away'],
        how='left'
    )
    df = df.drop(columns=[
        'game_id_home', 'team_home', 'game_id_away', 'team_away'
    ])
    elo_features = compute_elo_ratings(
        games[['game_id', 'home_team', 'away_team', 'home_score', 'away_score', 'season', 'week']]
    )
    df = df.merge(elo_features, on='game_id', how='left')
    return df


def compute_elo_ratings(games: pd.DataFrame, base_rating: float = 1500.0, k_factor: float = 20.0) -> pd.DataFrame:
    """Compute pre‑game Elo ratings for each team.

    Elo ratings provide a running estimate of team strength. Ratings for each
    game are recorded before the game is played to avoid data leakage.
    """
    ratings: dict[str, float] = defaultdict(lambda: base_rating)
    records: list[dict[str, float]] = []
    for row in games.sort_values(['season', 'week']).itertuples():
        home = row.home_team
        away = row.away_team
        game_id = row.game_id
        home_rating = ratings[home]
        away_rating = ratings[away]
        records.append({'game_id': game_id, 'elo_home': home_rating, 'elo_away': away_rating})
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
            if row.home_score > row.away_score:
                actual_home = 1.0
            elif row.home_score < row.away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5
            margin = abs(row.home_score - row.away_score)
            rating_diff = abs(home_rating - away_rating)
            mult = (margin + 1.0) / (rating_diff + 1.0)
            k_adjusted = k_factor * mult
            new_home_rating = home_rating + k_adjusted * (actual_home - expected_home)
            new_away_rating = away_rating + k_adjusted * ((1.0 - actual_home) - (1.0 - expected_home))
            ratings[home] = new_home_rating
            ratings[away] = new_away_rating
    return pd.DataFrame(records)


class NegativeBinomialRegressor:
    """Sklearn‑style wrapper around statsmodels Negative Binomial regression."""
    def __init__(self) -> None:
        self.result = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if sm is None:
            raise ImportError("statsmodels is required for NegativeBinomialRegressor")
        X_const = sm.add_constant(X, has_constant='add')
        model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())
        self.result = model.fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Model has not been fitted yet")
        X_const = sm.add_constant(X, has_constant='add')
        return self.result.predict(X_const)


def prepare_model_data(games_with_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare feature matrix X and target matrix Y for modelling."""
    df = games_with_stats.copy()
    numeric_features = [
        'offense_avg_last5_home', 'defense_avg_last5_home',
        'offense_avg_season_home', 'defense_avg_season_home',
        'win_rate_last5_home',
        'offense_avg_last5_away', 'defense_avg_last5_away',
        'offense_avg_season_away', 'defense_avg_season_away',
        'win_rate_last5_away',
        'elo_home', 'elo_away',
    ]
    for col in ['spread_line', 'total_line']:
        if col in df.columns:
            numeric_features.append(col)
            df[col] = df[col].fillna(df[col].mean())
    advanced_feature_candidates = [
        'red_zone_offense_rate', 'red_zone_defense_rate',
        'third_down_offense_rate', 'third_down_defense_rate',
        'penalty_yards', 'turnover_count',
        'off_epa', 'def_epa',
        'off_success_rate', 'def_success_rate',
        'pace', 'run_pass_ratio', 'no_huddle_rate',
        'rest_days', 'travel_distance', 'time_zone_change',
        'altitude', 'temperature', 'wind_speed', 'precipitation',
        'stadium_indoor', 'qb_injury_indicator'
    ]
    for col in advanced_feature_candidates:
        if col in df.columns:
            numeric_features.append(col)
    categorical_features = ['season', 'week', 'home_team', 'away_team']
    X_numeric = df[numeric_features].copy()
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=False)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X = X.fillna(X.mean())
    Y = df[['home_score', 'away_score']].copy()
    return X, Y


def evaluate_models(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate several regression algorithms."""
    models: dict[str, MultiOutputRegressor] = {
        'LinearRegression': MultiOutputRegressor(LinearRegression()),
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        'Poisson': MultiOutputRegressor(
            PoissonRegressor(alpha=0.01, max_iter=300)
        ),
        'ElasticNet': MultiOutputRegressor(
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        ),
        'NeuralNetwork': MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(64,),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            )
        ),
    }
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            model.fit(X_train, Y_train)
            Y_pred = pd.DataFrame(
                model.predict(X_test),
                index=Y_test.index,
                columns=['home_pred', 'away_pred']
            )
            y_true_all.append(Y_test.reset_index(drop=True))
            y_pred_all.append(Y_pred.reset_index(drop=True))
        y_true_concat = pd.concat(y_true_all, ignore_index=True)
        y_pred_concat = pd.concat(y_pred_all, ignore_index=True)
        mae_home = mean_absolute_error(y_true_concat['home_score'], y_pred_concat['home_pred'])
        mae_away = mean_absolute_error(y_true_concat['away_score'], y_pred_concat['away_pred'])
        rmse_home = np.sqrt(mean_squared_error(y_true_concat['home_score'], y_pred_concat['home_pred']))
        rmse_away = np.sqrt(mean_squared_error(y_true_concat['away_score'], y_pred_concat['away_pred']))
        r2_home = r2_score(y_true_concat['home_score'], y_pred_concat['home_pred'])
        r2_away = r2_score(y_true_concat['away_score'], y_pred_concat['away_pred'])
        mae_total = mae_home + mae_away
        rmse_total = rmse_home + rmse_away
        results.append({
            'model': name,
            'MAE_home': mae_home,
            'MAE_away': mae_away,
            'MAE_total': mae_total,
            'RMSE_home': rmse_home,
            'RMSE_away': rmse_away,
            'RMSE_total': rmse_total,
            'R2_home': r2_home,
            'R2_away': r2_away,
        })
    return pd.DataFrame(results)


def train_final_model(X: pd.DataFrame, Y: pd.DataFrame, model_name: str = 'ElasticNet'):
    """Train a final model on all available data."""
    model_mapping: dict[str, MultiOutputRegressor] = {
        'LinearRegression': MultiOutputRegressor(LinearRegression()),
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        'Poisson': MultiOutputRegressor(
            PoissonRegressor(alpha=0.01, max_iter=300)
        ),
        'ElasticNet': MultiOutputRegressor(
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        ),
        'NeuralNetwork': MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(64,),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            )
        ),
        **({'NegativeBinomial': MultiOutputRegressor(NegativeBinomialRegressor())} if sm is not None else {})
    }
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model name: {model_name}")
    model = model_mapping[model_name]
    model.fit(X, Y)
    return model


def predict_future_games(model, games_with_stats: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Generate score predictions for upcoming games.

    Selects the latest week in the provided dataset as the prediction window.
    Returns a dataframe with predicted scores, spread, and total.
    """
    latest_season = games_with_stats['season'].max()
    latest_week = games_with_stats.loc[games_with_stats['season'] == latest_season, 'week'].max()
    upcoming_mask = (games_with_stats['season'] == latest_season) & (games_with_stats['week'] == latest_week)
    upcoming_games = games_with_stats[upcoming_mask].copy()
    X_upcoming = X.loc[upcoming_games.index]
    preds = model.predict(X_upcoming)
    upcoming_games['home_pred'] = preds[:, 0]
    upcoming_games['away_pred'] = preds[:, 1]
    upcoming_games['pred_spread'] = upcoming_games['home_pred'] - upcoming_games['away_pred']
    upcoming_games['pred_total'] = upcoming_games['home_pred'] + upcoming_games['away_pred']
    for col in ['spread_line', 'total_line']:
        if col in games_with_stats.columns:
            upcoming_games[col] = games_with_stats.loc[upcoming_games.index, col]
        else:
            upcoming_games[col] = np.nan
    return upcoming_games[['season', 'week', 'home_team', 'away_team',
                           'home_pred', 'away_pred',
                           'pred_spread', 'pred_total',
                           'spread_line', 'total_line']]


def main() -> None:
    """Demonstrate training and prediction for a specified season/week."""
    start_year = 2015
    prediction_season = 2025
    prediction_week = 1
    print(f"Loading NFL schedules from {start_year}–{prediction_season} (including unplayed games)...")
    games_full = load_game_data(start_year, prediction_season, include_unplayed=True)
    print("Computing rolling team statistics...")
    games_with_stats_full = compute_team_statistics(games_full)
    print("Preparing model inputs...")
    X_full, Y_full = prepare_model_data(games_with_stats_full)
    train_mask = (
        games_with_stats_full['home_score'].notna() &
        (
            (games_with_stats_full['season'] < prediction_season) |
            ((games_with_stats_full['season'] == prediction_season) & (games_with_stats_full['week'] < prediction_week))
        )
    )
    pred_mask = (
        (games_with_stats_full['season'] == prediction_season) &
        (games_with_stats_full['week'] == prediction_week)
    )
    X_train = X_full.loc[train_mask].reset_index(drop=True)
    Y_train = Y_full.loc[train_mask].reset_index(drop=True)
    X_pred = X_full.loc[pred_mask].copy()
    if len(X_train) > 0:
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        X_pred = X_pred.fillna(train_means)
    else:
        X_train = X_train.fillna(0)
        X_pred = X_pred.fillna(0)
    if len(X_train) > 0:
        print("Evaluating candidate models using time‑series cross‑validation on training data...")
        results = evaluate_models(X_train, Y_train)
        print("\nModel comparison results (training data):")
        print(results.to_string(index=False))
        best_model_name = results.sort_values('RMSE_total').iloc[0]['model']
    else:
        print("No historical games available for training; defaulting to ElasticNet model.")
        best_model_name = 'ElasticNet'
    print(f"\nTraining final model: {best_model_name}")
    final_model = train_final_model(X_train, Y_train, model_name=str(best_model_name))
    print(f"\nGenerating score predictions for season {prediction_season}, week {prediction_week}...")
    preds = final_model.predict(X_pred)
    upcoming_games = games_with_stats_full.loc[pred_mask, ['season', 'week', 'home_team', 'away_team']].copy()
    upcoming_games['home_pred'] = preds[:, 0]
    upcoming_games['away_pred'] = preds[:, 1]
    upcoming_games['pred_spread'] = upcoming_games['home_pred'] - upcoming_games['away_pred']
    upcoming_games['pred_total'] = upcoming_games['home_pred'] + upcoming_games['away_pred']
    for col in ['spread_line', 'total_line']:
        if col in games_with_stats_full.columns:
            upcoming_games[col] = games_with_stats_full.loc[pred_mask, col].values
        else:
            upcoming_games[col] = np.nan
    print(upcoming_games)
    output_filename = f'score_predictions_{prediction_season}_wk{prediction_week}.csv'
    upcoming_games.to_csv(output_filename, index=False)
    print(f"Predictions have been saved to '{output_filename}'.")


if __name__ == '__main__':
    main()
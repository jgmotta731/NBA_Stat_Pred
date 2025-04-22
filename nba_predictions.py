# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:15:53 2025

@author: jgmot
"""

# predict_nba_bins.py
import os, joblib, warnings, sys
from datetime import date, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from nba_api.stats.static import players
from unidecode import unidecode
from nba_model import NBAStatPredictor
from torch.serialization import add_safe_globals

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------
# Load & Prepare Data
# ---------------------------------------------------
gamelogs = pd.read_parquet('nba_gamelogs.parquet')
# downcast numeric types
gamelogs[gamelogs.select_dtypes('float64').columns] = \
    gamelogs.select_dtypes('float64')\
            .apply(pd.to_numeric, downcast='float')
gamelogs[gamelogs.select_dtypes('int64').columns] = \
    gamelogs.select_dtypes('int64')\
            .apply(pd.to_numeric, downcast='integer')
gamelogs = gamelogs[gamelogs['did_not_play'] == False].copy()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
rolling_cols = [
    'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
    'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
    'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'steals', 'blocks',
    'turnovers', 'fouls', 'points', 'minutes'
]
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']

gamelogs.dropna(subset=rolling_cols + targets, inplace=True)
gamelogs.sort_values(by=['season', 'game_date', 'team_abbreviation', 'athlete_display_name'], inplace=True)

gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(int)
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])
gamelogs['days_since_last_game'] = gamelogs.groupby('athlete_display_name')['game_date'].diff().dt.days

# Rolling averages: 3, 5, 10 games back, shifted
for window in [3, 5, 10]:
    for col in rolling_cols:
        gamelogs[f'{col}_rolling{window}'] = (
            gamelogs.groupby('athlete_display_name')[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

# Lag team winner: 1 to 6 games back
for lag in range(1, 6):
    gamelogs[f'team_winner_lag{lag}'] = gamelogs.groupby('athlete_display_name')['team_winner'].shift(lag)

# Lag binary categorical columns: 1 to 3 games back
for col in ['is_playoff', 'ejected', 'starter']:
    for lag in range(1, 4):
        gamelogs[f'{col}_lag{lag}'] = gamelogs.groupby('athlete_display_name')[col].shift(lag)

# Opponent rolling defense metrics
gamelogs.sort_values(by=['season', 'game_date', 'opponent_team_abbreviation'], inplace=True)
for w in [3, 5, 10]:
    # Rolling averages of the opponent’s own score
    gamelogs[f"opponent_team_score_rolling{w}"] = (
        gamelogs
          .groupby("opponent_team_abbreviation")["opponent_team_score"]
          .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )

    # Rolling averages of points the opponent allowed
    gamelogs[f"opponent_points_allowed_{w}"] = (
        gamelogs
          .groupby("opponent_team_abbreviation")["team_score"]
          .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    )

    # Rank the opponent’s defence within the season for this window
    gamelogs[f"opponent_defense_rank_{w}"] = (
        gamelogs
          .groupby("season")[f"opponent_points_allowed_{w}"]
          .rank(method="average", pct=True)     # lower points‑allowed ⟹ lower (better) percentile
    )

# Choose one window (10‑game here) as the “main” defence‑rank column
gamelogs["opponent_defense_rank"] = gamelogs["opponent_defense_rank_10"]

# -----------------------------------------------------------------------------
# 3) Label‑encode IDs using your saved encoders
# -----------------------------------------------------------------------------
player_le   = joblib.load("player_encoder.pkl")
team_le     = joblib.load("team_encoder.pkl")
opponent_le = joblib.load("opponent_encoder.pkl")

gamelogs['player_id']   = player_le.transform(gamelogs['athlete_display_name'])
gamelogs['team_id']     = team_le.transform(gamelogs['team_abbreviation'])
gamelogs['opponent_id'] = opponent_le.transform(gamelogs['opponent_team_abbreviation'])

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------
gamelogs.drop(columns=[
    'minutes', 'plus_minus', 'field_goals_made', 'field_goals_attempted',
    'three_point_field_goals_attempted', 'turnovers', 'fouls',
    'free_throws_attempted', 'free_throws_made', 'starter', 'ejected',
    'did_not_play', 'team_winner', 'team_score', 'opponent_team_score', 'active',
    'season_type'
], inplace=True)

available_cols = set(gamelogs.columns)
lagged_rolling_features = [
    f'{col}_lag{i}'
    for col in rolling_cols for i in [1, 2, 3]
    if f'{col}_lag{i}' in available_cols
]

features = (
    ['home_away', 'athlete_position_abbreviation']
    # player‑centric rolling means
    + [f'{col}_rolling{w}' for col in rolling_cols for w in [3, 5, 10]]
    # binary‑flag lags
    + [f'{col}_lag{i}' for col in ['ejected', 'starter', 'is_playoff'] for i in [1, 2, 3]]
    # numeric stat lags that already exist
    + lagged_rolling_features
    # team‑winner lags
    + [f'team_winner_lag{i}' for i in range(1, 6)]
    # opponent‑defence windows (3, 5, 10)
    + [f'opponent_team_score_rolling{w}'   for w in [3, 5, 10]]
    + [f'opponent_points_allowed_{w}'      for w in [3, 5, 10]]
    + [f'opponent_defense_rank_{w}'        for w in [3, 5, 10]]
    # keep the plain alias column for whichever window you chose (10‑game in our example)
    + ['opponent_defense_rank']
    # remaining contextual / ID features
    + ['is_playoff', 'days_since_last_game',
       'player_id', 'team_id', 'opponent_id']
)

gamelogs.dropna(subset=features + targets, inplace=True)

# -----------------------------------------------------------------------------
# 5) Build X_full exactly as in training
# -----------------------------------------------------------------------------
X_raw     = gamelogs[features].copy()
X_num     = X_raw.select_dtypes(include=[np.number])
X_cat     = X_raw.select_dtypes(include=['object','category'])
X_encoded = pd.get_dummies(X_cat)            # same one‐hot logic
X_full    = pd.concat([X_num, X_encoded],axis=1)

# If your training set had columns that don’t appear here (or vice versa),
# make sure you reindex to the *trained* column set:
# trained_cols = joblib.load("trained_feature_columns.pkl")
# X_full = X_full.reindex(columns=trained_cols, fill_value=0)

# -----------------------------------------------------------------------------
# 6) Load each bin‐model JSON, predict_proba, and append probs
# -----------------------------------------------------------------------------
manual_bins = {
    'points':        [0,5,10,15,20,25,30,40,60],
    'assists':       [0,2,4,6,8,10,15],
    'rebounds':      [0,3,6,9,12,15,20],
    'steals':        [0,1,2,3,4,5],
    'blocks':        [0,1,2,3,4,5],
    'three_point_field_goals_made':[0,1,2,3,4,6,8]
}

for target in manual_bins:
    mdl = xgb.XGBClassifier()
    # direct filename—no path needed
    mdl.load_model(f"bin_model_{target}_bin.json")
    probs = mdl.predict_proba(X_full)  # shape = (n_rows, n_bins)
    for i in range(probs.shape[1]):
        gamelogs[f"{target}_prob_{i}"] = probs[:,i]

# -----------------------------------------------------------------------------
# 7) Features
# -----------------------------------------------------------------------------
embedding_features = ['player_id','team_id','opponent_id']

categorical_features = [
    'home_away','athlete_position_abbreviation','is_playoff',
    'ejected_lag1','starter_lag1','is_playoff_lag1',
    'ejected_lag2','starter_lag2','is_playoff_lag2',
    'ejected_lag3','starter_lag3','is_playoff_lag3',
    'team_winner_lag1','team_winner_lag2','team_winner_lag3',
    'team_winner_lag4','team_winner_lag5'
]

numeric_features = [c for c in gamelogs.columns.tolist() if c not in categorical_features + embedding_features]

add_safe_globals([NBAStatPredictor])
model        = torch.load("nba_model_full.pt", weights_only=False)   # ← loaded once
model.eval()
preprocessor = joblib.load("preprocessor.pkl")
# ---------------------------------------------------

# ---------------------------------------------------
# Predict Next Games (Only 2025 starters)
# ---------------------------------------------------
latest_players = (
    gamelogs[gamelogs['season'] == 2025]
    .sort_values('game_date')
    .groupby('athlete_display_name')
    .tail(1)
)
latest_players = latest_players[latest_players['starter_lag1'] == True].copy()

schedule = pd.read_parquet("nba_schedule.parquet")
schedule_clean = schedule[
    (schedule['home_abbreviation'] != 'TBD') &
    (schedule['away_abbreviation'] != 'TBD')
].copy()

schedule_clean['home_away'] = 'home'
schedule_away = schedule_clean.copy()
schedule_away['home_away'] = 'away'
schedule_away[['home_abbreviation','away_abbreviation']] = \
    schedule_away[['away_abbreviation','home_abbreviation']]
schedule_expanded = pd.concat([schedule_clean, schedule_away], ignore_index=True)
schedule_expanded.rename(columns={
    'home_abbreviation': 'team_abbreviation',
    'away_abbreviation': 'opponent_team_abbreviation'
}, inplace=True)

schedule_players = schedule_expanded.merge(
    latest_players,
    on='team_abbreviation',
    how='left',
    suffixes=('','_player')
)
schedule_players.drop(columns=[
    'game_date_player','home_away_player','opponent_team_abbreviation_player'
], errors='ignore', inplace=True)
schedule_players.dropna(subset=features, inplace=True)

# --- ensure the schedule dates are proper datetimes ---
schedule_players['game_date'] = pd.to_datetime(
    schedule_players['game_date'], errors='coerce'
)

# build model inputs
X_pred          = preprocessor.transform(
    schedule_players[numeric_features + categorical_features]
)
player_ids_pred = schedule_players['player_id'].values
team_ids_pred   = schedule_players['team_id'].values
opp_ids_pred    = schedule_players['opponent_id'].values

# make predictions
with torch.no_grad():
    preds = model(
        torch.tensor(X_pred, dtype=torch.float32),
        torch.tensor(player_ids_pred, dtype=torch.long),
        torch.tensor(team_ids_pred, dtype=torch.long),
        torch.tensor(opp_ids_pred, dtype=torch.long)
    )
    preds = torch.expm1(preds).numpy()

# assemble results
pred_df = schedule_players[[
    'athlete_display_name','athlete_position_abbreviation',
    'team_abbreviation','opponent_team_abbreviation',
    'game_date','home_away'
]].copy()

for i, col in enumerate(targets):
    pred_df[f'predicted_{col}'] = preds[:, i].round(1)

# --- filter to current week using Timestamps ---
today = date.today()
start_of_week = pd.to_datetime(today - timedelta(days=today.weekday()))
end_of_week   = start_of_week + timedelta(days=6)

pred_df = pred_df[
    (pred_df['game_date'] >= start_of_week) &
    (pred_df['game_date'] <= end_of_week)
]
pred_df['game_date'] = pred_df['game_date'].dt.strftime('%Y-%m-%d')

# add headshots & save
player_df = pd.DataFrame(players.get_active_players())
player_df['headshot_url'] = player_df['id'].apply(
    lambda pid: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
)

player_df['full_name'] = player_df['full_name'].apply(lambda name: unidecode(name).title())
pred_df['normalized_name'] = pred_df['athlete_display_name'].apply(lambda name: unidecode(name).title())

pred_df = pred_df.merge(
    player_df,
    left_on='normalized_name',
    right_on='full_name',
    how='left'
)

pred_df.drop(columns=['id', 'full_name', 'first_name', 'last_name', 'is_active', 'normalized_name'], inplace=True)
pred_df['home_away'] = pred_df['home_away'].str.capitalize()

pred_df.to_parquet("nba_predictions.parquet", index=False)
print("Predictions saved to nba_predictions.parquet")

print("Python exe :", sys.executable)
print("Conda env  :", os.getenv("CONDA_DEFAULT_ENV"))
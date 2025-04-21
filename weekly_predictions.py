"""
Created on Sun Apr 20 2025

@author: jgmot
"""

# ---------------------------------------------------
# Imports & Seed Setup
# ---------------------------------------------------
import os, random, warnings, json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.static import players
from nba_model import NBAStatPredictor
from torch.serialization import add_safe_globals

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# ---------------------------------------------------
# Label Encoding
# ---------------------------------------------------
player_le = LabelEncoder()
team_le = LabelEncoder()
opponent_le = LabelEncoder()
gamelogs['player_id']   = player_le.fit_transform(gamelogs['athlete_display_name'])
gamelogs['team_id']     = team_le.fit_transform(gamelogs['team_abbreviation'])
gamelogs['opponent_id'] = opponent_le.fit_transform(gamelogs['opponent_team_abbreviation'])

# Save encoders
joblib.dump(player_le, 'player_encoder.pkl')
joblib.dump(team_le, 'team_encoder.pkl')
joblib.dump(opponent_le, 'opponent_encoder.pkl')

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

# ---------------------------------------------------
# XGBoost Prob Bins
# ---------------------------------------------------
manual_bins = {
    'points':        [0, 5, 10, 15, 20, 25, 30, 40, 60],
    'assists':       [0, 2, 4, 6, 8, 10, 15],
    'rebounds':      [0, 3, 6, 9, 12, 15, 20],
    'steals':        [0, 1, 2, 3, 4, 5],
    'blocks':        [0, 1, 2, 3, 4, 5],
    'three_point_field_goals_made': [0, 1, 2, 3, 4, 6, 8]
}

def make_manual_bins(df, manual_bins):
    for target, bins in manual_bins.items():
        if bins[-1] != float('inf'):
            bins = bins + [float('inf')]
        df[f'{target}_bin'] = pd.cut(
            df[target], bins=bins, labels=False, include_lowest=True
        )
    return df

gamelogs = make_manual_bins(gamelogs, manual_bins)
bin_target_cols = [f'{t}_bin' for t in targets]

# prepare classification dataset
X_classify = gamelogs[features].copy()
X_classify_encoded = pd.get_dummies(X_classify.select_dtypes(include=['object', 'category']))
X_classify_numeric = X_classify.select_dtypes(include=[np.number])
X_full = pd.concat([X_classify_numeric, X_classify_encoded], axis=1)
y_classify = gamelogs[bin_target_cols]

models = {}
for target in y_classify.columns:
    clf = xgb.XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss',
        use_label_encoder=False, n_estimators=500, early_stopping_rounds=20,
        learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
        tree_method='hist', random_state=SEED, n_jobs=6
    )
    clf.fit(X_full, y_classify[target], eval_set=[(X_full, y_classify[target])], verbose=True)
    models[target] = clf

# ---------------------------------------------------
# Predict Probabilities & Add to Gamelogs
# ---------------------------------------------------
probability_features = []
for target in y_classify.columns:
    prob_array = models[target].predict_proba(X_full)
    for i in range(prob_array.shape[1]):
        col_name = f"{target.replace('_bin','')}_prob_{i}"
        gamelogs[col_name] = prob_array[:, i]
        features.append(col_name)
        probability_features.append(col_name)

# drop bin columns and any remaining NaNs
gamelogs.drop(columns=bin_target_cols, inplace=True)
gamelogs.dropna(inplace=True)

# ---------------------------------------------------
# ▶▶ NEW BLOCK — define feature groups & load artifacts
# ---------------------------------------------------
embedding_features = ['player_id','team_id','opponent_id']

categorical_features = [
    'home_away','athlete_position_abbreviation','is_playoff',
    'ejected_lag1','starter_lag1','is_playoff_lag1',
    'ejected_lag2','starter_lag2','is_playoff_lag2',
    'ejected_lag3','starter_lag3','is_playoff_lag3',
    'team_winner_lag1','team_winner_lag2','team_winner_lag3',
    'team_winner_lag4','team_winner_lag5'
]

numeric_features = [c for c in features if c not in categorical_features + embedding_features]

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
pred_df = pred_df.merge(
    player_df,
    left_on='athlete_display_name',
    right_on='full_name',
    how='left'
)

pred_df.to_csv("nba_predictions.csv", index=False)
print("Predictions saved to nba_predictions.csv")


import sys, os
print("Python exe :", sys.executable)
print("Conda env  :", os.getenv("CONDA_DEFAULT_ENV"))

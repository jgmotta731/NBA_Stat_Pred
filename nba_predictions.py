# ---------------------------------------------------
# NBA Predictions Pipeline for New Data
# Created on Apr 29, 2025
# Author: Jack Motta
# ---------------------------------------------------

import os, gc, joblib, warnings
import numpy as np
import pandas as pd
import re
from datetime import date, timedelta
from unidecode import unidecode
from nba_api.stats.static import players
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

preprocessor = joblib.load("preprocessor_pipeline.joblib")
selector = joblib.load("variance_threshold_selector.joblib")
multi_task_lasso = joblib.load("multi_task_lasso_model.joblib")
multi_rf_classifier = joblib.load("multi_rf_classifier_model.joblib")
multi_rf_regressor = joblib.load("multi_rf_regressor_model.joblib")

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

gamelogs = pd.read_parquet("nba_gamelogs.parquet")
schedule = pd.read_parquet("nba_schedule.parquet")
player_clusters = pd.read_parquet("player_clusters.parquet")

# ---------------------------------------------------
# Preprocessing New Gamelogs
# ---------------------------------------------------

# Downcast
gamelogs[gamelogs.select_dtypes('float64').columns] = \
    gamelogs.select_dtypes('float64')\
            .apply(pd.to_numeric, downcast='float')
gamelogs[gamelogs.select_dtypes('int64').columns] = \
    gamelogs.select_dtypes('int64')\
            .apply(pd.to_numeric, downcast='integer')

# Only keep 2022 season onwards
gamelogs = gamelogs[gamelogs["season"] >= 2022].copy()

# Response variables
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']

# Drop games with missing target stats (i.e., not played)
gamelogs = gamelogs.dropna(subset=targets).copy()

# Count games per player
player_game_counts = gamelogs.groupby("athlete_display_name").size()

# Keep only players with >=10 valid games
valid_players = player_game_counts[player_game_counts >= 10].index
gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

# After filtering players ➔ filter low-minute games
gamelogs = gamelogs[gamelogs["minutes"] >= 20].reset_index(drop=True)

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

# Convert booleans to numeric
for col in ['starter', 'ejected', 'team_winner', 'is_playoff']:
    if col in gamelogs.columns:
        gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)

rolling_cols = [
    'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
    'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
    'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'steals', 'blocks',
    'turnovers', 'fouls', 'points', 'minutes'
]

gamelogs = gamelogs.sort_values(by=['season', 'game_date', 'team_abbreviation', 'athlete_display_name'])

gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])
gamelogs['days_since_last_game'] = gamelogs.groupby('athlete_display_name')['game_date'].diff().dt.days.astype(np.float32)

# Rolling Metrics: 10, shifted
for window in [10]:
    for col in rolling_cols:
        # Rolling Averages
        gamelogs[f'{col}_rolling{window}'] = (
            gamelogs.groupby('athlete_display_name')[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        ).astype(np.float32)

        # Momentum
        gamelogs[f'{col}_momentum{window}'] = (
            gamelogs.groupby('athlete_display_name')[col]
            .transform(lambda x: x.shift(1)) - 
            gamelogs.groupby('athlete_display_name')[col]
            .transform(lambda x: x.shift(2).rolling(window, min_periods=1).mean())
        ).astype(np.float32)

        # Global rolling rank
        gamelogs[f'{col}_global_rolling_rank{window}'] = (
            gamelogs.groupby('game_date')[f'{col}_rolling{window}']
            .rank(method="average", pct=True)
        ).astype(np.float32)

    # Efficiency ratios
    gamelogs[f'assist_to_turnover_rolling{window}'] = (
        gamelogs[f'assists_rolling{window}'] / (gamelogs[f'turnovers_rolling{window}'] + 1)
    ).astype(np.float32)

    gamelogs[f'orb_pct_rolling{window}'] = (
        gamelogs[f'offensive_rebounds_rolling{window}'] / gamelogs[f'rebounds_rolling{window}']
    ).astype(np.float32)

    gamelogs[f'fg_pct_rolling{window}'] = (
        gamelogs[f'field_goals_made_rolling{window}'] / gamelogs[f'field_goals_attempted_rolling{window}']
    ).astype(np.float32)

    gamelogs[f'ft_pct_rolling{window}'] = (
        gamelogs[f'free_throws_made_rolling{window}'] / gamelogs[f'free_throws_attempted_rolling{window}']
    ).astype(np.float32)

for window in [10]:
    # Opponent Defense
    gamelogs[f"opponent_team_score_rolling{window}"] = (
        gamelogs.groupby("opponent_team_abbreviation")["opponent_team_score"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ).astype(np.float32)

    gamelogs[f"opponent_points_allowed_rolling{window}"] = (
        gamelogs.groupby("opponent_team_abbreviation")["team_score"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ).astype(np.float32)

    gamelogs[f"opponent_defense_rank_rolling{window}"] = (
        gamelogs.groupby("season")[f"opponent_points_allowed_rolling{window}"]
        .rank(method="average", pct=True)
    ).astype(np.float32)

    # Team Offense
    gamelogs[f"team_score_rolling{window}"] = (
        gamelogs.groupby("team_abbreviation")["team_score"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ).astype(np.float32)

    gamelogs[f"team_offense_rank_rolling{window}"] = (
        gamelogs.groupby("season")[f"team_score_rolling{window}"]
        .rank(method="average", pct=True)
    ).astype(np.float32)

# Rolling Win Streak
gamelogs['rolling_win_streak'] = (
    gamelogs.sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['team_winner']
    .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
).astype(np.int32)

# Rolling Loss Streak
gamelogs['rolling_loss_streak'] = (
    gamelogs.sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['team_winner']
    .transform(lambda x: x.shift(1).groupby((x.shift(1) != 0).cumsum()).cumcount())
).astype(np.int32)

# Safe division (clip small denominators)
def safe_divide(numerator, denominator, eps=1e-3):
    return numerator / (denominator + eps)

gamelogs[f'fg_pct_rolling{window}'] = safe_divide(
    gamelogs[f'field_goals_made_rolling{window}'],
    gamelogs[f'field_goals_attempted_rolling{window}']
).clip(0, 1).astype(np.float32)

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------

gamelogs = gamelogs.drop(columns=[
    'minutes', 'plus_minus', 'field_goals_made', 'field_goals_attempted',
    'three_point_field_goals_attempted', 'turnovers', 'fouls',
    'free_throws_attempted', 'free_throws_made', 'ejected',
    'did_not_play', 'team_winner', 'team_score', 'opponent_team_score', 'active',
    'season_type', 'starter'
    ])

available_cols = set(gamelogs.columns)

lagged_rolling_features = [col for col in gamelogs.columns
                           if 'rolling' in col or 'lag' in col]

numeric_features = lagged_rolling_features + ['days_since_last_game']
categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff"]
features = numeric_features + categorical_features

# Ensure uniqueness in lists
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))
features = list(dict.fromkeys(features))

# Add missing flag
def add_generic_missing_flag(X_df, feature_list):
    X_df = X_df.copy()
    X_df['was_missing'] = X_df[feature_list].isna().any(axis=1).astype(int)
    return X_df

gamelogs = add_generic_missing_flag(gamelogs, features)

# Final recomputed feature groups
features = numeric_features + ['was_missing'] + categorical_features

# Enforce uniqueness
features = list(dict.fromkeys(features))
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))

# Merge cluster labels
gamelogs = gamelogs.merge(player_clusters, on='athlete_display_name', how='left')

# Convert int32 to float32
gamelogs[gamelogs.select_dtypes('int32').columns] = gamelogs.select_dtypes('int32').astype('float32')

# ---------------------------------------------------
# Match Gamelogs to Schedule
# ---------------------------------------------------

latest_players = (
    gamelogs[gamelogs['season'] == 2025]
    .sort_values('game_date')
    .groupby('athlete_display_name')
    .tail(1)
)

schedule_clean = schedule[(schedule['home_abbreviation'] != 'TBD') & (schedule['away_abbreviation'] != 'TBD')].copy()
schedule_clean['home_away'] = 'home'
schedule_away = schedule_clean.copy()
schedule_away['home_away'] = 'away'
schedule_away[['home_abbreviation', 'away_abbreviation']] = schedule_away[['away_abbreviation', 'home_abbreviation']]
schedule_expanded = pd.concat([schedule_clean, schedule_away], ignore_index=True)
schedule_expanded.rename(columns={
    'home_abbreviation': 'team_abbreviation',
    'away_abbreviation': 'opponent_team_abbreviation'
}, inplace=True)

schedule_players = schedule_expanded.merge(
    latest_players,
    on='team_abbreviation',
    how='left'
)
schedule_players.dropna(subset=['athlete_display_name'], inplace=True)
schedule_players = schedule_players.drop(columns=[col for col in schedule_players.columns if col.endswith('_y')])
schedule_players.columns = [col[:-2] if col.endswith('_x') else col for col in schedule_players.columns]
schedule_players['game_date'] = pd.to_datetime(schedule_players['game_date'], errors='coerce')

# ---------------------------------------------------
# Generate Model Inputs
# ---------------------------------------------------

model_features = features + ['cluster_label']

X_pred = preprocessor.transform(schedule_players[model_features])
X_pred = selector.transform(X_pred)

new_features = joblib.load("selected_features.joblib")
selected_indices = joblib.load("selected_indices.joblib")

# Stage 1: MultiTask Lasso predictions
y_pred_lasso = multi_task_lasso.predict(X_pred)
X_pred_selected = X_pred[:, selected_indices]
X_pred_stacked = np.hstack([X_pred_selected, y_pred_lasso])

# Stage 2: MultiOutput Classifier predictions
y_pred_class = multi_rf_classifier.predict(X_pred_stacked)
X_pred_stacked = np.hstack([X_pred_stacked, y_pred_class])

# Final Stage: MultiOutput Regressor predictions
y_pred_final = multi_rf_regressor.predict(X_pred_stacked)

# ---------------------------------------------------
# Assemble Final Predictions
# ---------------------------------------------------

pred_df = schedule_players[[
    'athlete_display_name','athlete_position_abbreviation',
    'team_abbreviation','opponent_team_abbreviation',
    'game_date','home_away'
]].copy()

# Clip negative predictions
pred_cols = ['predicted_three_point_field_goals_made', 'predicted_rebounds', 'predicted_assists',
             'predicted_steals', 'predicted_blocks', 'predicted_points']

predictions_df = pd.DataFrame(y_pred_final, columns=pred_cols)
pred_df = pd.concat([pred_df.reset_index(drop=True), predictions_df], axis=1)

# Filter only for games in the current week
today = date.today()
start_of_week = pd.to_datetime(today - timedelta(days=today.weekday()))
end_of_week = start_of_week + timedelta(days=6)

pred_df = pred_df[(pred_df['game_date'] >= start_of_week) & (pred_df['game_date'] <= end_of_week)]
pred_df = pred_df.sort_values('game_date').drop_duplicates('athlete_display_name', keep='first')
pred_df['game_date'] = pred_df['game_date'].dt.strftime('%Y-%m-%d')

# Merge headshots
try:
    player_df = pd.DataFrame(players.get_active_players())
    player_df['headshot_url'] = player_df['id'].apply(
        lambda pid: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
    )
    player_df['full_name'] = player_df['full_name'].apply(lambda name: unidecode(name).title())
    pred_df['normalized_name'] = pred_df['athlete_display_name'].apply(lambda name: unidecode(name).title())
    pred_df = pred_df.merge(player_df, left_on='normalized_name', right_on='full_name', how='left')
    pred_df.drop(columns=['id', 'full_name', 'first_name', 'last_name', 'is_active', 'normalized_name'], inplace=True)
except Exception as e:
    print("Warning: could not fetch player headshots:", e)

# Clip negatives
pred_df[pred_cols] = pred_df[pred_cols].clip(lower=0)

# Remove all punctuation (including periods) from athlete_display_name
pred_df['athlete_display_name'] = pred_df['athlete_display_name'].str.replace(r'[^\w\s]', '', regex=True)
betting_odds = pd.read_csv('Pivoted_Betting_Odds.csv')
betting_odds['player_name'] = betting_odds['player_name'].str.replace(r'[^\w\s]', '', regex=True)

# Only keep player_name column
betting_odds = betting_odds[['player_name']].copy()

# Inner join ➜ keep only players in both prediction and betting data
pred_df = pred_df.merge(
    betting_odds,
    left_on='athlete_display_name',
    right_on='player_name',
    how='inner'
)


pred_df = pred_df.drop(columns='player_name')

# Save
pred_df.to_parquet("nba_predictions.parquet", index=False)
print("Predictions saved to nba_predictions.parquet")

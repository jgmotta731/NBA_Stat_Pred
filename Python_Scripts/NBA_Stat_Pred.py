# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 2025
@author: jgmot
"""
# ---------------------------------------------------
# Import Modules
# ---------------------------------------------------
from __future__ import annotations
import os
import re
import time
import math
import types
import random
import warnings
import unicodedata
from typing import List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoreadvancedv2, leaguegamelog
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    root_mean_squared_error, r2_score, mean_absolute_error, mean_pinball_loss,
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import colorcet as cc

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
pd.set_option('display.max_columns', 20)
os.environ["PYTENSOR_FLAGS"] = "cxx=g++"

# ---------------------------------------------------
# Load & Prepare Data
# ---------------------------------------------------
# Scrape Game Metadata for joining purposes
def get_game_metadata(seasons, delay=1.0, save_path="datasets/game_metadata.parquet"):
    all_logs = []
    for season in seasons:
        for season_type in ["Regular Season", "Playoffs"]:
            try:
                time.sleep(delay)  # Prevent API throttling
                log_df = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    timeout=10  # ⏱️ Add timeout here

                ).get_data_frames()[0]
                log_df = log_df[['GAME_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP']].drop_duplicates()
                log_df['SEASON'] = season
                log_df['SEASON_TYPE'] = season_type
                all_logs.append(log_df)
            except Exception as e:
                print(f"Failed to load {season} ({season_type}): {e}")
    if all_logs:
        metadata_df = pd.concat(all_logs, ignore_index=True)
        metadata_df.to_parquet(save_path, index=False)
        print(f"Game metadata saved to: {save_path}")
        return metadata_df
    else:
        print("No metadata retrieved.")
        return None

# Run on post-COVID seasons
seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
game_metadata_df = get_game_metadata(seasons)

# -----------------------------
# Config
# -----------------------------
CHUNK_SIZE = 120
CHUNK_SLEEP = 420  # 7 minutes
PER_REQUEST_SLEEP = (2.0, 4.0)

# -----------------------------
# Fetch Advanced Boxscore with retry
# -----------------------------
def get_advanced_boxscore(game_id, retries=2, timeout=10, backoff=2.5):
    attempt = 0
    while attempt < retries:
        try:
            time.sleep(random.uniform(*PER_REQUEST_SLEEP))
            box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=timeout)
            df = box.get_data_frames()[0]
            df['GAME_ID'] = game_id
            return df
        except Exception as e: 
            print(f"Error for {game_id} on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(backoff * attempt)
    return None

# -----------------------------
# Process a Chunked Season
# -----------------------------
def process_season_in_chunks(season, metadata_df, output_dir="datasets/season_parquet_chunks", overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"advanced_boxscore_{season.replace('-', '_')}.parquet")
    # Load previously saved chunk (if exists)
    existing_df = pd.read_parquet(outfile) if os.path.exists(outfile) and not overwrite else pd.DataFrame()
    all_game_ids = metadata_df.loc[metadata_df["SEASON"] == season, "GAME_ID"].unique()
    already_done = set(existing_df["GAME_ID"]) if not existing_df.empty else set()
    remaining_ids = [gid for gid in all_game_ids if gid not in already_done]
    print(f"Processing {len(remaining_ids)} remaining games for {season}...")
    collected_data = [existing_df] if not existing_df.empty else []
    for i in range(0, len(remaining_ids), CHUNK_SIZE):
        chunk = remaining_ids[i:i + CHUNK_SIZE]
        chunk_data = []
        print(f"Starting chunk {i//CHUNK_SIZE + 1} of {season}...")
        for game_id in tqdm(chunk, desc=f"{season} - Chunk {i//CHUNK_SIZE + 1}"):
            df = get_advanced_boxscore(game_id)
            if df is not None:
                df["SEASON"] = season
                chunk_data.append(df)
        if chunk_data:
            chunk_df = pd.concat(chunk_data, ignore_index=True)
            collected_data.append(chunk_df)
            # Save progressively
            combined_df = pd.concat(collected_data, ignore_index=True)
            combined_df.to_parquet(outfile, index=False)
            print(f"Chunk saved to {outfile}")
        # Cooldown
        print(f"Sleeping for {CHUNK_SLEEP / 60:.1f} minutes to avoid throttling...")
        time.sleep(CHUNK_SLEEP)
    final_df = pd.read_parquet(outfile)
    print(f"Finished season {season}: {len(final_df)} games saved.")
    return final_df

# Run for multiple seasons
seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]

# Initialize list to collect all season data
all_dfs = []

# Process each season
for season in seasons:
    df = process_season_in_chunks(season, metadata_df=game_metadata_df)
    if df is not None:
        all_dfs.append(df)

# Combine and save all seasons if data was collected
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_parquet("datasets/nba_advanced_boxscores_all_seasons.parquet", index=False)
    print("All seasons combined and saved.")
else:
    print("No data collected.")

# Merge advanced box score data with game metadata
combined_df_merged = combined_df.merge(
    game_metadata_df,
    on=["GAME_ID", "TEAM_ABBREVIATION"],
    how="left"
)

# Save the merged dataset
combined_df_merged.to_parquet("datasets/nba_advanced_boxscores_with_metadata.parquet", index=False)
print("Merged data saved to nba_advanced_boxscores_with_metadata.parquet")

# ------------------ Preprocessing Checkpoint ------------------
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made',
                'free_throws_attempted', 'free_throws_made',
                'three_point_field_goals_attempted'] + targets

def normalize_name(name):
    name = name.strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\b(?:[IVX]+)$', '', name)
    return name.strip()
    
READY_FOR_FE_PATH = "datasets/gamelogs_ready_for_fe.parquet"

if os.path.exists(READY_FOR_FE_PATH):
    print("Using cached gamelogs (assembly stage)")
    gamelogs = pd.read_parquet(READY_FOR_FE_PATH)
else:
    print("Assembling gamelogs")

    # ------------------ ORIGINAL PIPELINE ------------------
    # Read Parquet
    combined_df_merged = pd.read_parquet("datasets/nba_advanced_boxscores_with_metadata.parquet")

    # Drop NAs in advanced stats
    combined_df_merged = combined_df_merged.dropna().reset_index(drop=True)

    injury_db = pd.concat([
        pd.read_csv('Injury Database.csv'),
        pd.read_csv('2025_injuries.csv')
    ], ignore_index=True)

    gamelogs = pd.read_parquet('datasets/nba_gamelogs.parquet')
    lineups = pd.read_parquet('datasets/lineup_stints.parquet')
    onoff = pd.read_parquet('datasets/onoff_player_game.parquet')

    gamelogs = gamelogs.dropna().reset_index(drop=True)
    gamelogs[gamelogs.select_dtypes('float64').columns] = gamelogs.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
    gamelogs[gamelogs.select_dtypes('int64').columns] = gamelogs.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')

    injury_db['norm'] = injury_db['PLAYER'].apply(normalize_name)
    gamelogs['norm'] = (
        gamelogs['athlete_display_name']
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\b(?:[IVX]+)$', '', regex=True)
        .str.strip()
    )

    injury_db['DATE'] = pd.to_datetime(injury_db['DATE'], errors='coerce')
    gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'], errors='coerce')
    combined_df_merged['GAME_DATE'] = pd.to_datetime(combined_df_merged['GAME_DATE'], errors='coerce')

    injury_db = injury_db[injury_db['DATE'].dt.year >= 2021].copy()
    gamelogs = gamelogs[gamelogs['game_date'].dt.year >= 2021].copy().reset_index(drop=True)

    starter_norms = set(gamelogs.loc[gamelogs['starter'], 'norm'].unique())
    starter_injuries = injury_db[injury_db['norm'].isin(starter_norms)].copy()
    starter_injuries['starter_injured'] = True
    injured_team_dates = starter_injuries[['TEAM', 'DATE', 'starter_injured']].drop_duplicates(subset=['TEAM', 'DATE'])

    gamelogs = gamelogs.merge(injured_team_dates,
                               left_on=['team_display_name', 'game_date'],
                               right_on=['TEAM', 'DATE'],
                               how='left')
    gamelogs['starter_injured'] = gamelogs['starter_injured'].fillna(False)

    combined_df_merged['norm'] = combined_df_merged['PLAYER_NAME'].apply(normalize_name)
    gamelogs = gamelogs.merge(combined_df_merged,
                               how='left',
                               left_on=['game_date', 'team_abbreviation', 'norm'],
                               right_on=['GAME_DATE', 'TEAM_ABBREVIATION', 'norm'])

    gamelogs = gamelogs[gamelogs["norm"].isin(combined_df_merged["norm"])].copy()
    gamelogs = gamelogs.drop(columns=[
        'SEASON_x', 'SEASON_y', 'TEAM_ABBREVIATION', 'GAME_DATE', 'SEASON_TYPE',
        'norm', 'COMMENT', 'MIN', 'NICKNAME', 'START_POSITION', 'PLAYER_NAME',
        'PLAYER_ID', 'TEAM_CITY', 'TEAM_ID', 'GAME_ID', 'TEAM', 'DATE', 'MATCHUP',
        'team_display_name'
    ], errors='ignore')

    gamelogs = gamelogs.dropna(subset=targets2).copy()
    player_game_counts = gamelogs.groupby("athlete_display_name").size()
    valid_players = player_game_counts[player_game_counts >= 10].index
    gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

    gamelogs = gamelogs[gamelogs["did_not_play"] == False].reset_index(drop=True)
    gamelogs = gamelogs[gamelogs["minutes"] > 0].reset_index(drop=True)

    gamelogs = gamelogs.sort_values(['athlete_display_name', 'game_date', 'season',
                                     'team_abbreviation']).reset_index(drop=True)

    qualified_keys = (
        gamelogs
        .groupby(['athlete_display_name', 'season'])['minutes']
        .mean()
        .reset_index()
        .query('minutes >= 15')[['athlete_display_name', 'season']]
    )

    gamelogs = gamelogs[
        gamelogs[['athlete_display_name', 'season']].apply(tuple, axis=1).isin(
            qualified_keys.apply(tuple, axis=1)
        )
    ]

    recent_season = gamelogs['season'].max()
    current_players = gamelogs[gamelogs['season'] == recent_season]['athlete_display_name'].unique()
    gamelogs = gamelogs[gamelogs['athlete_display_name'].isin(current_players)]

    gamelogs[gamelogs.select_dtypes('float64').columns] = gamelogs.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
    gamelogs[gamelogs.select_dtypes('int64').columns] = gamelogs.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
    gamelogs.columns = gamelogs.columns.str.lower()

    lineups['player_ids'] = lineups['lineup'].str.split('-')
    exploded = lineups.explode('player_ids').copy()
    exploded['player_id'] = exploded['player_ids'].astype(int)
    exploded['points_per_min'] = exploded['points'] / exploded['duration_minutes'].replace(0, np.nan)

    agg_stats = (
        exploded.groupby(['player_id', 'game_id'])
        .agg(
            total_lineup_minutes=('duration_minutes', 'sum'),
            num_unique_lineups=('lineup', 'nunique')
        )
        .reset_index()
    )

    weighted_avg = (
        exploded.groupby(['player_id', 'game_id'])
        .apply(lambda df: np.average(df['points'], weights=df['duration_minutes']))
        .reset_index(name='avg_lineup_ppm')
    )

    player_lineup_features = pd.merge(agg_stats, weighted_avg, on=['player_id', 'game_id'])

    onoff['player_id'] = onoff['player_id'].astype('int32')
    onoff = onoff.drop_duplicates(subset=['player_id', 'game_id'])

    gamelogs = gamelogs.merge(onoff, on=['player_id', 'game_id'], how='left')
    gamelogs = gamelogs.merge(player_lineup_features, on=['player_id', 'game_id'], how='left')

    # ------------------ SAVE CHECKPOINT ------------------
    gamelogs.to_parquet(READY_FOR_FE_PATH, index=False)
    print("Saved gamelogs_ready_for_fe.parquet")
    
# Function to delete unnecessary objects from memory excluding modules and specified objects
def del_except(*keep):
    # Classes to delete (pandas/numpy objects)
    deletable_types = (
        pd.DataFrame,
        pd.Series,
        pd.Index,
        np.ndarray,
        list,
        dict,
        set,
        tuple,
        int,
        float,
        str
    )
    keep = set(keep) | {'del_except'}
    for name, val in list(globals().items()):
        if name.startswith('_') or name in keep:
            continue
        if isinstance(val, types.ModuleType):
            continue
        if isinstance(val, (types.FunctionType, type)):
            continue
        if isinstance(val, deletable_types):
            try:
                del globals()[name]
            except Exception:
                pass
            
# Clean up
del_except('gamelogs', 'SEED', 'targets', 'targets2')

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
FEATURED_PATH = "datasets/gamelogs_features.parquet"

if os.path.exists(FEATURED_PATH):
    print("Using cached feature-engineered gamelogs")
    gamelogs = pd.read_parquet(FEATURED_PATH)

else:
    print("Running full feature engineering pipeline")
    
    # Ensure game_date is datetime
    gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])
    
    # Drop All-Star Games
    all_star_teams = ['DUR', 'LEB', 'GIA', 'WEST', 'EAST', 'CAN', 'CHK', 'KEN',
                      'SHQ']
    gamelogs = gamelogs[~gamelogs['team_abbreviation'].isin(all_star_teams)].reset_index(drop=True)
    
    # Ensure sorting for cumulative logic
    gamelogs = gamelogs.sort_values(['opponent_team_abbreviation', 'season', 'game_date']).reset_index(drop=True)
    
    # Opponent cumulative losses: team_winner == 1 → opponent lost
    gamelogs['opp_cum_losses'] = (
        (gamelogs['team_winner'] == 1)
        .groupby([gamelogs['opponent_team_abbreviation'], gamelogs['season']])
        .cumsum()
        .astype(np.int32)
    )
    
    # Opponent cumulative wins: team_winner == 0 → opponent won
    gamelogs['opp_cum_wins'] = (
        (gamelogs['team_winner'] == 0)
        .groupby([gamelogs['opponent_team_abbreviation'], gamelogs['season']])
        .cumsum()
        .astype(np.int32)
    )
    
    # Opponent Win %
    gamelogs['opp_win_pct'] = gamelogs['opp_cum_wins'] / (gamelogs['opp_cum_wins'] + gamelogs['opp_cum_losses'])
    
    # Opponent Days Since Last Game
    gamelogs['opp_days_since_last_game'] = (
        gamelogs
        .groupby(['opponent_team_abbreviation', 'season'])['game_date']
        .diff()
        .dt.days
        .astype(np.float32)
    )
    
    # Opponent Back-to-Back Flag
    gamelogs['opp_is_back_to_back'] = (gamelogs['opp_days_since_last_game'] == 1).astype(np.int32)
    
    # Create base opponent games table
    opponent_games = (
        gamelogs[['opponent_team_abbreviation', 'season', 'game_date']]
        .drop_duplicates()
        .sort_values(['opponent_team_abbreviation', 'season', 'game_date'])
        .copy()
    )
    
    # Compute number of opponent games in the past 7 days using rolling window
    opponent_games['opp_games_last_7d'] = (
        opponent_games
        .groupby(['opponent_team_abbreviation', 'season'], group_keys=False)
        .apply(lambda group: pd.Series(
            group.set_index('game_date')
                 .rolling('7D', on=None)['opponent_team_abbreviation']  # count games
                 .count()
                 .values,
            index=group.index
        ))
        .astype(np.float32)
    )
    
    # Merge cleanly back
    gamelogs = gamelogs.merge(
        opponent_games,
        on=['opponent_team_abbreviation', 'season', 'game_date'],
        how='left'
    )
    
    # Global temporal sort
    gamelogs = gamelogs.sort_values(['athlete_display_name', 'game_date', 'season',
                                     'team_abbreviation']).reset_index(drop=True)
    
    # Number of teammates injured
    gamelogs['num_starters_injured'] = gamelogs.groupby(['game_date', 'team_abbreviation'])['starter_injured'].transform('sum')
    
    # Clean plus_minus
    if 'plus_minus' in gamelogs.columns:
        gamelogs['plus_minus'] = (
            gamelogs['plus_minus']
            .astype(str)
            .str.replace(r'^\+', '', regex=True)
            .replace('None', np.nan)
            .astype(np.float32))
    
    # Ensure 'starter' is numeric (1 for starter, 0 for not)
    gamelogs['starter'] = gamelogs['starter'].astype(int)
    
    # Booleans to numeric
    bool_cols = ['starter', 'ejected', 'team_winner', 'is_playoff', 'starter_injured']
    for col in bool_cols:
        if col in gamelogs.columns:
            gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)
    
    # Playoff indicator
    gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)
    
    # Days since Last game
    gamelogs['days_since_last_game'] = gamelogs.groupby(['athlete_display_name', 'season'])['game_date'].diff().dt.days.astype(np.float32)
    
    # Back-to-back
    gamelogs['is_back_to_back'] = (gamelogs['days_since_last_game'].fillna(99) == 1).astype(np.int32)
    
    # Calculate FG%, 3P%, and FT%
    gamelogs['fg_pct'] = (
        gamelogs['field_goals_made'] / gamelogs['field_goals_attempted'].replace(0, np.nan)
    ).astype(float)
    
    gamelogs['fg3_pct'] = (
        gamelogs['three_point_field_goals_made'] / gamelogs['three_point_field_goals_attempted'].replace(0, np.nan)
    ).astype(float)
    
    gamelogs['ft_pct'] = (
        gamelogs['free_throws_made'] / gamelogs['free_throws_attempted'].replace(0, np.nan)
    ).astype(float)
    
    span = 21 # Set span for reactive f-string
    
    # Columns to compute rolling/expanding/ewm on
    rolling_cols = [
        # Core performance
        'minutes', 'field_goals_made', 'field_goals_attempted',
        'three_point_field_goals_made', 'three_point_field_goals_attempted',
        'free_throws_made', 'free_throws_attempted', 'fg_pct', 'fg3_pct', 'ft_pct',
        'rebounds', 'assists', 'steals', 'blocks', 'points',
        # Advanced team context
        'off_rating', 'def_rating', 'ast_pct', 'oreb_pct', 'dreb_pct', 'reb_pct',
        'efg_pct', 'ts_pct', 'usg_pct', 'pace', 'poss',
        # On/Off Columns
        'on_points', 'on_fga', 'on_fgm', 'on_fg3a', 'on_fg3m', 
        'on_offensive_rebounds', 'on_defensive_rebounds',
        'on_assists', 'on_turnovers', 'off_points', 'off_fga', 'off_fgm',
        'off_fg3a', 'off_fg3m', 'off_offensive_rebounds',
        'off_defensive_rebounds', 'off_assists', 'off_turnovers',
        # Lineup Columns
        'total_lineup_minutes', 'num_unique_lineups', 'avg_lineup_ppm'
    ]
    
    lag_cols = rolling_cols + ['ejected']
    
    # Lags Function
    def compute_lag_features(df, col):
        df = df.sort_values(['athlete_display_name', 'game_date'])
        result = pd.DataFrame(index=df.index)
        result[f'{col}_lag1'] = df.groupby(['athlete_display_name'])[col].shift(1)
        result[f'{col}_lag2'] = df.groupby(['athlete_display_name'])[col].shift(2)
        return result
    
    lag_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_lag_features)(gamelogs, col) for col in lag_cols
    )
    gamelogs = pd.concat([gamelogs] + lag_results, axis=1)
    
    def compute_expanding_features(df, col):
        df = df.sort_values(['athlete_display_name', 'season', 'game_date'])
        result = pd.DataFrame(index=df.index)
        shifted = df.groupby(['athlete_display_name'])[col].shift(1)
        grouped = shifted.groupby([df['athlete_display_name']])
        result[f'{col}_expanding_mean'] = grouped.expanding().mean().reset_index(level=[0, 1], drop=True)
        result[f'{col}_expanding_std'] = grouped.expanding().std().reset_index(level=[0, 1], drop=True)
        result[f'{col}_expanding_max'] = grouped.expanding().max().reset_index(level=[0, 1], drop=True)
        result[f'{col}_expanding_min'] = grouped.expanding().min().reset_index(level=[0, 1], drop=True)
        temp = df[['game_date']].copy()
        temp[f'{col}_expanding_mean'] = result[f'{col}_expanding_mean']
        result[f'{col}_expanding_mean_rank'] = (
            temp.groupby('game_date')[f'{col}_expanding_mean']
                .transform(lambda x: x.rank(pct=True))
                .astype(np.float32)
        )
        return result
    
    # Apply in parallel
    expanding_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_expanding_features)(gamelogs, col) for col in rolling_cols
    )
    
    # Concatenate to main DataFrame
    gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)
    
    # Rolling Window Function
    def compute_rolling_features(df, col, window=5):
        df = df.sort_values(['athlete_display_name', 'game_date']).copy()
        result = pd.DataFrame(index=df.index)
        shifted = df.groupby('athlete_display_name')[col].shift(1)
        grouped = shifted.groupby(df['athlete_display_name'])
        rolling_mean = grouped.rolling(window).mean().reset_index(level=0, drop=True)
        result[f'{col}_rolling_mean'] = rolling_mean
        result[f'{col}_rolling_std'] = grouped.rolling(window).std().reset_index(level=0, drop=True)
        result[f'{col}_rolling_max'] = grouped.rolling(window).max().reset_index(level=0, drop=True)
        result[f'{col}_rolling_min'] = grouped.rolling(window).min().reset_index(level=0, drop=True)
        result[f'{col}_rolling_mad'] = grouped.rolling(window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).reset_index(level=0, drop=True)
        result[f'{col}_rolling_trend'] = grouped.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=False
        ).reset_index(level=0, drop=True)
        # Rolling mean rank per game date
        df_temp = df[['game_date']].copy()
        df_temp[f'{col}_rolling_mean'] = rolling_mean
        result[f'{col}_rolling_mean_rank'] = (
            df_temp.groupby('game_date')[f'{col}_rolling_mean']
            .transform(lambda x: x.rank(pct=True).astype(np.float32))
        )
        return result
    
    # Apply in parallel
    rolling_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_rolling_features)(gamelogs, col, window=5) for col in rolling_cols
    )
    
    # Add to gamelogs
    gamelogs = pd.concat([gamelogs] + rolling_results, axis=1)
    
    # EWM Features
    def compute_ewm_features(df, col):
        df = df.sort_values(['athlete_display_name', 'game_date'])
        result = pd.DataFrame(index=df.index)
        ewm_spans = [21]
        for span in ewm_spans:
            shifted = df.groupby('athlete_display_name')[col].shift(1)
            grouped = shifted.groupby(df['athlete_display_name'])
            ewm_mean = grouped.ewm(span=span, adjust=False).mean().reset_index(level=0, drop=True)
            ewm_std = grouped.ewm(span=span, adjust=False).std().reset_index(level=0, drop=True)
            result[f'{col}_ewm_mean_span{span}'] = ewm_mean
            result[f'{col}_ewm_std_span{span}'] = ewm_std
            result[f'{col}_ewm_zscore_span{span}'] = (shifted - ewm_mean) / (ewm_std + 1e-6)
            result[f'{col}_ewm_above_avg_span{span}'] = (shifted > ewm_mean).astype(int)
            surge = (shifted > ewm_mean).astype(int)
            result[f'{col}_ewm_surge_count_span{span}'] = (
                surge.groupby(df['athlete_display_name']).cumsum())
            result[f'{col}_ewm_reversion_score_span{span}'] = (
                (shifted - ewm_mean).abs() / (ewm_std + 1e-6))
            # Approximate volatility via absolute deviation from EWM mean
            result[f'{col}_ewm_mad_span{span}'] = grouped.transform(lambda x: np.abs(x - x.ewm(span=span, adjust=False).mean())).reset_index(level=0, drop=True)
            # EWM rank per game date
            temp_ewm = df[['game_date']].copy()
            temp_ewm[f'{col}_ewm_mean_span{span}'] = ewm_mean
            result[f'{col}_ewm_mean_span{span}_rank'] = (
                temp_ewm.groupby('game_date')[f'{col}_ewm_mean_span{span}']
                    .transform(lambda x: x.rank(pct=True))
                    .astype(np.float32)
            )
        return result
    
    # Apply in parallel
    ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_ewm_features)(gamelogs, col) for col in rolling_cols
    )
    
    # Add to gamelogs
    gamelogs = pd.concat([gamelogs] + ewm_results, axis=1)
    
    def compute_delta_features(df, col):
        result = pd.DataFrame(index=df.index)
        # Difference between lag and expanding baseline
        result[f'{col}_delta_lag1_vs_expanding'] = (
            df[f'{col}_lag1'] - df[f'{col}_expanding_mean']
        )
        return result
    
    # Parallel apply delta features
    delta_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_delta_features)(gamelogs, col) for col in rolling_cols
    )
    
    # Concatenate delta features
    gamelogs = pd.concat([gamelogs] + delta_results, axis=1)
    
    # Home/Away Shooting Splits
    shooting_cols = ['points', 'field_goals_made', 'field_goals_attempted',
                     'three_point_field_goals_made', 'three_point_field_goals_attempted']
    
    def compute_home_and_away_ewm(df, col):
        result = pd.DataFrame(index=df.index)
        span = 21
        for context in ['home', 'away']:
            df_sub = df[df['home_away'] == context].copy()
            df_sub = df_sub.sort_values(['athlete_display_name', 'season', 'game_date'])
            # Shift by 1 to prevent leakage
            shifted = df_sub.groupby('athlete_display_name')[col].shift(1)
            # EWM calculation
            ewm_mean = shifted.groupby(df_sub['athlete_display_name']) \
                .ewm(span=span, adjust=False).mean().reset_index(level=0, drop=True)
            ewm_col = f'{col}_ewm_mean_{context}_span{span}'
            # Store EWM means
            result.loc[df_sub.index, ewm_col] = ewm_mean
            # Compute rank per game_date
            rank_col = f'{ewm_col}_rank'
            result.loc[df_sub.index, rank_col] = (
                ewm_mean.groupby(df_sub['game_date'])
                .transform(lambda x: x.rank(pct=True).astype(np.float32))
            )
        return result
    
    homeaway_ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_home_and_away_ewm)(gamelogs, col)
        for col in shooting_cols
    )
    
    gamelogs = pd.concat([gamelogs] + homeaway_ewm_results, axis=1)
    
    # Function for EWM Trend Slope via linear regression
    def compute_ewm_trend_stats_for_player(name, group, col, span):
        group = group.sort_values('game_date')
        values = group[col].shift(1).to_numpy()
        idx = group.index.to_numpy()
        slopes = np.full(len(values), np.nan, dtype=np.float32)
        for i in range(span - 1, len(values)):
            y = values[:i + 1]
            x = np.arange(len(y)).reshape(-1, 1)
            mask = ~np.isnan(y)
            if np.count_nonzero(mask) >= 3 and not np.allclose(y[mask], y[mask][0]):
                try:
                    weights = pd.Series(np.ones(len(y))).ewm(span=span, adjust=False).mean().to_numpy()
                    weights = weights[mask]
                    x_masked = x[mask]
                    y_masked = y[mask]
                    model = LinearRegression()
                    model.fit(x_masked, y_masked, sample_weight=weights)
                    slopes[i] = model.coef_[0]
                except:
                    pass
        return idx, slopes
    
    # Compute trend slope in parallel
    for span in [21]:
        for col in rolling_cols:
            groups = gamelogs.groupby(['athlete_display_name', 'season'])
            results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
                delayed(compute_ewm_trend_stats_for_player)(name, grp, col, span)
                for name, grp in groups)
            slope_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)
            for idxs, slopes in results:
                slope_arr[idxs] = slopes
            slope_col = f'{col}_ewm_trend_slope_{span}'
            gamelogs[slope_col] = slope_arr
            # Add percentile rank of the slope per game date
            rank_col = f'{slope_col}_rank'
            gamelogs[rank_col] = (
                gamelogs.groupby('game_date')[slope_col]
                .transform(lambda x: x.rank(pct=True).astype(np.float32))
            )
    
    # Hot/cold flags
    gamelogs['is_hot_game'] = ((gamelogs['points'] >= gamelogs['points_ewm_mean_span21'] + gamelogs['points_ewm_std_span21']).fillna(False).astype(int))
    gamelogs['is_cold_game'] = ((gamelogs['points'] < gamelogs['points_ewm_mean_span21'] - gamelogs['points_ewm_std_span21']).fillna(False).astype(int))
    
    # Hot/Cold Streaks
    gamelogs['hot_streak'] = (
        gamelogs
        .sort_values(['athlete_display_name', 'season', 'game_date'])
        .groupby(['athlete_display_name', 'season'])['is_hot_game']
        .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
        .astype(np.int32)
    )
    
    gamelogs['cold_streak'] = (
        gamelogs
        .sort_values(['athlete_display_name', 'season', 'game_date'])
        .groupby(['athlete_display_name', 'season'])['is_cold_game']
        .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
        .astype(np.int32)
    )
    
    # Drop temp columns
    gamelogs = gamelogs.drop(columns=['is_hot_game', 'is_cold_game'], axis=1)
    
    # Opponent stats to compute
    opponent_stats = ['team_score', 'rebounds', 'three_point_field_goals_made', 'turnovers']
    ewm_spans = [21]
    
    # Compute opponent-based EWM allowed stats
    def compute_opponent_ewm_allowed_mean_rank(stat, span):
        # Step 1: Aggregate total stat allowed by opponent per game
        opponent_daily = (
            gamelogs
            .groupby(['opponent_team_abbreviation', 'season', 'game_date'])[stat]
            .sum()
            .reset_index()
            .sort_values(['opponent_team_abbreviation', 'season', 'game_date']))
        # Step 2: Apply shifted EWM to avoid leakage
        ewm_col = f'opponent_ewm_{stat}_allowed_span{span}'
        opponent_daily[ewm_col] = (
            opponent_daily
            .groupby(['opponent_team_abbreviation', 'season'])[stat]
            .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean()))
        # Step 3: For each game date, compute percentile rank of the EWM stat across all opponents
        rank_col = f'{ewm_col}_pct_rank'
        opponent_daily[rank_col] = (
            opponent_daily
            .groupby('game_date')[ewm_col]
            .transform(lambda x: x.rank(pct=True).astype(np.float32)))
        return opponent_daily[['opponent_team_abbreviation', 'season', 'game_date', ewm_col, rank_col]]
    
    # Run in parallel for all stat/span combinations
    opponent_ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_opponent_ewm_allowed_mean_rank)(stat, span)
        for stat in opponent_stats
        for span in ewm_spans
    )
    
    # Merge each result back into gamelogs
    for result in opponent_ewm_results:
        gamelogs = gamelogs.merge(result, on=['opponent_team_abbreviation', 'season', 'game_date'], how='left')
    
    # Rename Opponent Points Allowed Column Accordingly
    gamelogs = gamelogs.rename(columns={
        'opponent_ewm_team_score_allowed_span21_pct_rank': 'opponent_ewm_points_allowed_span21_pct_rank',
        'opponent_ewm_team_score_allowed_span21': 'opponent_ewm_points_allowed_span21',
    })
    
    # Opponent FG and 3PT % Allowed
    def compute_opponent_ewm_shooting_pct_allowed_rank(span):
        # Step 1: Aggregate per opponent per game
        daily = (gamelogs.groupby(['opponent_team_abbreviation', 'season', 'game_date'])[
                ['field_goals_made', 'field_goals_attempted', 
                 'three_point_field_goals_made', 'three_point_field_goals_attempted'
                ]].sum().reset_index().sort_values(['opponent_team_abbreviation', 'season', 'game_date']))
        # Step 2: Shift to prevent leakage
        for made_col, att_col in [
            ('field_goals_made', 'field_goals_attempted'),
            ('three_point_field_goals_made', 'three_point_field_goals_attempted')
        ]:
            daily[f'{made_col}_shift'] = daily.groupby(['opponent_team_abbreviation', 'season'])[made_col].shift(1)
            daily[f'{att_col}_shift'] = daily.groupby(['opponent_team_abbreviation', 'season'])[att_col].shift(1)
        # Step 3: Compute EWM components
        daily['ewm_fgm'] = daily.groupby(['opponent_team_abbreviation', 'season'])['field_goals_made_shift'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean())
        daily['ewm_fga'] = daily.groupby(['opponent_team_abbreviation', 'season'])['field_goals_attempted_shift'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean())
        daily['ewm_3pm'] = daily.groupby(['opponent_team_abbreviation', 'season'])['three_point_field_goals_made_shift'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean())
        daily['ewm_3pa'] = daily.groupby(['opponent_team_abbreviation', 'season'])['three_point_field_goals_attempted_shift'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean())
        # Step 4: Compute EWM FG% and 3PT% allowed
        daily['fg_pct'] = daily['ewm_fgm'] / daily['ewm_fga']
        daily['fg3_pct'] = daily['ewm_3pm'] / daily['ewm_3pa']
        daily[f'opponent_ewm_fg_pct_allowed_span{span}'] = daily['fg_pct']
        daily[f'opponent_ewm_fg3_pct_allowed_span{span}'] = daily['fg3_pct']
        # Step 5: Compute percentile ranks by game_date
        fg_rank_col = f'opponent_ewm_fg_pct_allowed_rank_span{span}'
        fg3_rank_col = f'opponent_ewm_fg3_pct_allowed_rank_span{span}'
        daily[fg_rank_col] = daily.groupby('game_date')['fg_pct'].transform(lambda x: x.rank(pct=True).astype(np.float32))
        daily[fg3_rank_col] = daily.groupby('game_date')['fg3_pct'].transform(lambda x: x.rank(pct=True).astype(np.float32))
        return daily[[
            'opponent_team_abbreviation', 'season', 'game_date',
            f'opponent_ewm_fg_pct_allowed_span{span}',
            f'opponent_ewm_fg3_pct_allowed_span{span}',
            fg_rank_col, fg3_rank_col
        ]]
    
    # Run in parallel
    results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_opponent_ewm_shooting_pct_allowed_rank)(span) for span in ewm_spans
    )
    
    # Merge results back into gamelogs
    for result in results:
        gamelogs = gamelogs.merge(result, on=['opponent_team_abbreviation', 'season', 'game_date'], how='left')
    
    # Compute Interactions
    def compute_ewm_player_opponent_interactions(df, stat, span):
        results = pd.DataFrame(index=df.index)
        # Rank-based interaction
        player_feat = f"{stat}_ewm_mean_span{span}"
        opp_rank_feat = f"opponent_ewm_{stat}_allowed_span{span}_pct_rank"
        rank_interaction_feat = f"{player_feat}_x_opp_allowed_rank"
        if player_feat in df.columns and opp_rank_feat in df.columns:
            results[rank_interaction_feat] = df[player_feat] * df[opp_rank_feat]
        else:
            print(f"Skipping RANK interaction for {stat}: missing {player_feat} or {opp_rank_feat}")
        # Raw mean interaction
        opp_raw_feat = f"opponent_ewm_{stat}_allowed_span{span}"
        raw_interaction_feat = f"{player_feat}_x_opp_allowed_raw"
        if player_feat in df.columns and opp_raw_feat in df.columns:
            results[raw_interaction_feat] = df[player_feat] * df[opp_raw_feat]
        else:
            print(f"Skipping RAW interaction for {stat}: missing {player_feat} or {opp_raw_feat}")
        return results
    
    # Assuming these are valid player stats you've EWM'ed
    interaction_stats = ['points', 'rebounds', 'turnovers', 'three_point_field_goals_made']
    
    # Parallel compute for all stats
    interaction_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
        delayed(compute_ewm_player_opponent_interactions)(gamelogs, stat, span=21)
        for stat in interaction_stats
    )
    
    # Concatenate into gamelogs
    gamelogs = pd.concat([gamelogs] + interaction_results, axis=1)
    
    # For each team/game, find the player with the highest expanding average before the game
    gamelogs['team_primary_scorer'] = (
        gamelogs
        .groupby(['team_abbreviation', 'game_date'])['points_ewm_mean_span21']
        .transform(lambda x: x == x.max()).astype(int))
    
    # Drop W/L columns
    gamelogs = gamelogs.drop(columns=['opp_cum_wins', 'opp_cum_losses', 'opp_win_pct'])
    
    # Ensure sort
    gamelogs = gamelogs.sort_values(['team_abbreviation', 'athlete_display_name', 'season', 'game_date']).copy()
    
    # Recent Team Injuries Count
    gamelogs['team_starter_injuries_rolling5'] = (
        gamelogs
        .sort_values(['team_abbreviation', 'season', 'game_date'])
        .groupby(['team_abbreviation', 'season'])['starter_injured']
        .transform(lambda x: x.shift(1).rolling(5).sum())
        .astype(np.float32))
    
    # Sort gamelogs again
    gamelogs = gamelogs.sort_values(['athlete_display_name', 'game_date', 'season',
                                     'team_abbreviation']).reset_index(drop=True)
    
    # Create More Interactions
    def generate_manual_interactions(df):
        span = 21  # Set globally for all interactions
        interaction_pairs = [
            # High-signal EWM Mean Interactions
            (f'minutes_ewm_mean_span{span}', f'usg_pct_ewm_mean_span{span}'),
            (f'points_ewm_mean_span{span}', f'pace_ewm_mean_span{span}'),
            (f'fg_pct_ewm_mean_span{span}', f'usg_pct_ewm_mean_span{span}'),
            (f'rebounds_ewm_mean_span{span}', f'reb_pct_ewm_mean_span{span}'),
            (f'assists_ewm_mean_span{span}', f'ast_pct_ewm_mean_span{span}'),
            (f'steals_ewm_mean_span{span}', f'def_rating_ewm_mean_span{span}'),
            (f'blocks_ewm_mean_span{span}', f'def_rating_ewm_mean_span{span}'),
            (f'rebounds_ewm_mean_span{span}', f'reb_pct_ewm_mean_span{span}'),
            (f'points_ewm_mean_span{span}', f'opponent_ewm_points_allowed_span{span}'),
            # New z-score and mad interactions
            (f'points_ewm_zscore_span{span}', f'pace_ewm_mean_span{span}'),
            (f'points_ewm_mad_span{span}', f'pace_ewm_mean_span{span}'),
            (f'steals_ewm_zscore_span{span}', f'def_rating_ewm_mean_span{span}'),
            (f'three_point_field_goals_made_ewm_zscore_span{span}', f'opponent_ewm_three_point_field_goals_made_allowed_span{span}'),
            (f'rebounds_ewm_mad_span{span}', f'reb_pct_ewm_mean_span{span}'),
            (f'assists_ewm_zscore_span{span}', f'ast_pct_ewm_mean_span{span}')
        ]
        result = pd.DataFrame(index=df.index)
        for var1, var2 in interaction_pairs:
            if var1 in df.columns and var2 in df.columns:
                colname = f'{var1}_x_{var2}'
                result[colname] = df[var1] * df[var2]
            else:
                print(f"Skipping: Missing {var1} or {var2}")
        return result
    
    # Generate interaction features
    interaction_features = generate_manual_interactions(gamelogs)
    
    # Concatenate them back to gamelogs
    gamelogs = pd.concat([gamelogs, interaction_features], axis=1)
    
    # Downcast float64 to float32
    float_cols = gamelogs.select_dtypes(include=['float64']).columns
    gamelogs[float_cols] = gamelogs[float_cols].astype(np.float32)
    
    # Downcast int64 to int32
    int_cols = gamelogs.select_dtypes(include=['int64']).columns
    gamelogs[int_cols] = gamelogs[int_cols].astype(np.int32)
    
    # Force any inf or -inf to nan
    gamelogs = gamelogs.replace([np.inf, -np.inf], np.nan)
    
    # Save checkpoint
    gamelogs.to_parquet(FEATURED_PATH, index=False)
    print(f"Saved feature-engineered gamelogs to {FEATURED_PATH}")

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------
# Select predictors excluding expanding features
lagged_rolling_features = [
    col for col in gamelogs.columns
    if ('rolling' in col or 'expanding' in col or 'trend' in col or 'lag' in col 
        or 'streak' in col or 'span' in col or 'ewm' in col)
]

# Group features
prior_features = [
    col for col in gamelogs.columns
    if any(col == f"{tgt}_expanding_{stat}" for stat in ["mean", "std"] for tgt in targets)
]

numeric_features = [
    col for col in lagged_rolling_features + [
        'days_since_last_game',
        'num_starters_injured',
        'opp_games_last_7d',
        'opp_days_since_last_game'
    ] if col not in prior_features
]

categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff",
                        "starter_injured", "is_back_to_back", "opp_is_back_to_back",
                        "team_primary_scorer"]
embedding_features = ['athlete_display_name', 'team_abbreviation', 'opponent_team_abbreviation']

# Final list of predictors (excluding embeddings and priors)
features = numeric_features + categorical_features

# Add missing flag
def add_generic_missing_flag(X_df, feature_list):
    X_df = X_df.copy()
    X_df['was_missing'] = X_df[feature_list].isna().any(axis=1).astype(int)
    return X_df
gamelogs = add_generic_missing_flag(gamelogs, features)

# Add Missing Flag to Categorical Features
categorical_features += ['was_missing']

# Final recomputed feature groups
features += ['was_missing']

# Enforce uniqueness
features = list(dict.fromkeys(features))
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))

# Apply delete function, keeping the following:
del_except('gamelogs', 'features', 'numeric_features', 'categorical_features',
           'embedding_features', 'prior_features', 'targets', 'targets2', 'SEED', 
           'lagged_rolling_features')

# ---------------------------------------------------
# PCA + Clustering (with Temporal Train/Test Split)
# ---------------------------------------------------
# Targets to include with both mean and std
targets_with_both = [
    'field_goals_made'
]

# Targets to include with mean only
targets_mean_only = [
    'field_goals_attempted',
    'reb_pct',
    'ast_pct',
    'usg_pct',
    'poss',
    'three_point_field_goals_attempted'
]

# Targets to include with std only
targets_std_only = [
    'points',
    'assists',
    'rebounds',
    'steals',
    'blocks',
    'three_point_field_goals_made'
]

# Combine all selected targets
selected_targets = targets_with_both + targets_mean_only + targets_std_only

# Build stat column list (only those that exist in gamelogs)
stat_cols = []

for target in selected_targets:
    # Include mean only if in appropriate group
    if target in targets_with_both or target in targets_mean_only:
        col_mean = f"{target}_expanding_mean"
        if col_mean in gamelogs.columns:
            stat_cols.append(col_mean)
    
    # Include std only if in appropriate group
    if target in targets_with_both or target in targets_std_only:
        col_std = f"{target}_expanding_std"
        if col_std in gamelogs.columns:
            stat_cols.append(col_std)

# Temporal train split (avoid data leakage from future data)
train_gamelogs = gamelogs[gamelogs['season'] < 2025].copy()

# Re-check which stat_cols actually exist in train_gamelogs (not just gamelogs)
stat_cols = [col for col in stat_cols if col in train_gamelogs.columns]

# Use latest expanding stat values per player instead of aggregates
train_player_summary = (
    train_gamelogs
    .sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')
    .tail(1)
    .set_index('athlete_display_name')[stat_cols]
    .reset_index()
)

# Preprocessing pipeline: median impute + robust scale
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), stat_cols)
])

# Fit PCA only on training player summary
X_train_proc = preprocessor.fit_transform(train_player_summary[stat_cols])
pca = PCA(random_state=SEED)
X_train_pca = pca.fit_transform(X_train_proc)

# Scree plot
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# Elbow plot
explained = pca.explained_variance_
components = np.arange(1, len(explained) + 1)
plt.figure(figsize=(10,6))
plt.bar(components[:10], explained[:10], width=0.8)
plt.xlim(0, 11)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot (First 10 Components)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Fit PCA pipeline (just PC1 + PC2)
pca_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2, random_state=SEED))
])

X_cluster_train = pca_pipeline.fit_transform(train_player_summary[stat_cols])

# Extract PCA loadings
pca_model = pca_pipeline.named_steps['pca']
loadings = pca_model.components_
feature_names = pca_pipeline.named_steps['preprocessor'].get_feature_names_out(stat_cols)

# Top contributors to PC1
num_pcs_to_show = min(20, loadings.shape[0])
for i in range(num_pcs_to_show):
    pc_loadings = pd.Series(loadings[i], index=feature_names, name=f'PC{i+1}_Loading') \
                    .sort_values(key=abs, ascending=False)
    print(f"\nTop contributors to PC{i+1}:")
    print(pc_loadings, '\n')

# KMeans evaluation loop
inertias = []
silhouette_scores = []
k_values = range(2, 31)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    kmeans.fit(X_cluster_train)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_train, kmeans.labels_))

# Plot elbow and silhouette
fig, ax1 = plt.subplots(figsize=(12, 5))
color = 'tab:blue'
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia (Distortion)', color=color)
ax1.plot(k_values, inertias, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:orange')
ax2.plot(k_values, silhouette_scores, marker='s', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Elbow Method + Silhouette Score')
plt.tight_layout()
plt.show()

# Fit Finnal K-Means Model
kmeans_final = KMeans(n_clusters=5, random_state=SEED)
kmeans_final.fit(X_cluster_train)

# Assign clusters
train_player_summary['cluster_label'] = kmeans_final.labels_ + 1

# Group by cluster and compute means
cluster_means = (
    train_player_summary
    .groupby('cluster_label')[stat_cols]
    .mean()
    .round(2)  # Optional: round for cleaner display
    .sort_index()
)

# Display
print(cluster_means)

# Create DataFrame with PC1 and PC2
pca_2d_df = pd.DataFrame(X_cluster_train[:, :2], columns=['PC1', 'PC2'])
pca_2d_df['cluster_label'] = train_player_summary['cluster_label'].values

# Use the first 21 Glasbey colors for clear cluster separation
distinct_palette = cc.glasbey[:21]

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=pca_2d_df,
    x='PC1',
    y='PC2',
    hue='cluster_label',
    palette=distinct_palette,
    s=80,
    alpha=0.85,
    edgecolor='k'
)
plt.title('2D PCA Cluster Distribution (PC1 vs PC2)', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=13)
plt.ylabel('Principal Component 2', fontsize=13)
plt.legend(title='Cluster', bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Add Cluster Labels to Gamelogs
# ---------------------------------------------------
# Build a full player summary (using only features trained above)
full_player_summary = (
    gamelogs
    .sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')
    .tail(1)
    .set_index('athlete_display_name')[stat_cols]
    .reset_index()
)

# Transform with the pretrained pipeline and predict cluster labels
X_full_cluster = pca_pipeline.transform(full_player_summary[stat_cols])
full_player_summary['cluster_label'] = kmeans_final.predict(X_full_cluster) + 1

# Extract just name + cluster_label
player_clusters = full_player_summary[['athlete_display_name', 'cluster_label']].copy()
player_clusters['cluster_label'] = player_clusters['cluster_label'].astype('float32')

# Merge cluster labels into gamelogs before any train/val split
gamelogs = (
    gamelogs
    .drop(columns=['cluster_label'], errors='ignore')
    .merge(player_clusters, on='athlete_display_name', how='left')
)

# Update feature lists
features += ['cluster_label']
categorical_features += ['cluster_label']

# Convert int64 to float32 dtype
gamelogs[gamelogs.select_dtypes(include='int64').columns] = gamelogs.select_dtypes(include='int64').astype('int32')

# Clear Memory
del_except('gamelogs', 'features', 'categorical_features', 'numeric_features', 
           'kmeans_final', 'pca_pipeline', 'targets', 'targets2', 'SEED', 
           'embedding_features', 'prior_features')

# Save feature engineered gamelogs
gamelogs.to_parquet('datasets/gamelogs_ready_for_modeling.parquet')

# ---------------------------------------------------
# Train/Validation Split
# ---------------------------------------------------
gamelogs = pd.read_parquet('datasets/gamelogs_ready_for_modeling.parquet')
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made',
                'free_throws_attempted', 'free_throws_made',
                'three_point_field_goals_attempted'] + targets

# Select predictors excluding expanding features
lagged_rolling_features = [
    col for col in gamelogs.columns
    if ('rolling' in col or 'expanding' in col or 'trend' in col or 'lag' in col 
        or 'streak' in col or 'span' in col or 'ewm' in col)
]

# Group features
prior_features = [
    col for col in gamelogs.columns
    if any(col == f"{tgt}_expanding_{stat}" for stat in ["mean", "std"] for tgt in targets)
]

numeric_features = [
    col for col in lagged_rolling_features + [
        'days_since_last_game',
        'num_starters_injured',
        'opp_games_last_7d',
        'opp_days_since_last_game'
    ] if col not in prior_features
]

categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff",
                        "starter_injured", "is_back_to_back", "opp_is_back_to_back",
                        "team_primary_scorer", "was_missing", "cluster_label"]
embedding_features = ['athlete_display_name', 'team_abbreviation', 'opponent_team_abbreviation']

# Final list of predictors (excluding embeddings and priors)
features = numeric_features + categorical_features

# Ensure uniqueness in lists
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))
features = list(dict.fromkeys(features))

train_df = gamelogs[gamelogs["season"] < 2025].copy()
val_df = gamelogs[gamelogs["season"] >= 2025].copy()

# Force categorical columns to be string type for compatibility
for col in categorical_features:
    train_df[col] = train_df[col].astype(str)
    val_df[col]   = val_df[col].astype(str)
    
# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------
# Fit encoder on training data and map unseen values to -1
embed_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-1,
)
embed_encoder.fit(train_df[embedding_features])

# Build embedding sizes from train vocab (add 1 so -1 becomes 0)
embedding_sizes = []
for cats in embed_encoder.categories_:
    n_seen = len(cats)                     # train vocab size
    emb_dim = min(50, max(4, int(np.sqrt(n_seen + 1))))  # small, stable
    embedding_sizes.append((n_seen + 1, emb_dim))        # +1 for UNK index 0

# Transform → shift → clamp (unknown/missing → -1 → +1 → 0)
X_train_embed = embed_encoder.transform(train_df[embedding_features]).astype(np.int64) + 1
X_val_embed   = embed_encoder.transform(val_df[embedding_features]).astype(np.int64) + 1

for j, (num_embeddings, _) in enumerate(embedding_sizes):
    X_train_embed[:, j] = np.clip(X_train_embed[:, j], 0, num_embeddings - 1)
    X_val_embed[:, j]   = np.clip(X_val_embed[:, j],   0, num_embeddings - 1)

# Quick diagnostics
na_rate = {embedding_features[j]: float((X_val_embed[:, j] == 0).mean())
            for j in range(len(embedding_features))}
print("Unknown rate on validation embeddings:", na_rate)

# ---------------------------------------------------
# Create Train and Test Sets
# ---------------------------------------------------
X_train = train_df[features]
X_val = val_df[features]

# Initialize scaler
y_scaler = StandardScaler()

# Fit on training target and transform both train and validation
y_train_scaled = y_scaler.fit_transform(train_df[targets])
y_val_scaled = y_scaler.transform(val_df[targets])

# ---------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------
# Define prior pipeline
prior_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
    ('scaler', StandardScaler())
])

# Fit and transform priors
prior_train_scaled = prior_pipeline.fit_transform(train_df[prior_features])
prior_val_scaled = prior_pipeline.transform(val_df[prior_features])

# Define Feature Pipelines
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32))
])

# Standard numeric transformer
standard_numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # retains 95% of variance
])

# Final ColumnTransformer
column_transformer = ColumnTransformer([
    ('num', standard_numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# Wrap into full preprocessing pipeline
preprocessor = Pipeline([
    ('transform', column_transformer),
    ('var_thres', VarianceThreshold(threshold=1e-4))
])

# Fit and transform
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

# Build final DataFrame with new feature names
X_train_proc = pd.DataFrame(X_train_proc, columns=preprocessor.get_feature_names_out())
X_val_proc = pd.DataFrame(X_val_proc, columns=preprocessor.get_feature_names_out())

# ---------- 0) Build breakout thresholds/labels for the 6 primaries ----------
break_targets = list(targets)  # exactly the main 6

# Per-row thresholds: τ = expanding_mean + expanding_std (assumes no leakage upstream)
tau_train = np.column_stack([
    train_df[f"{t}_expanding_mean"].values + train_df[f"{t}_expanding_std"].values
    for t in break_targets
]).astype("float32")
tau_val = np.column_stack([
    val_df[f"{t}_expanding_mean"].values + val_df[f"{t}_expanding_std"].values
    for t in break_targets
]).astype("float32")

# Raw labels for those same 6 stats (unscaled)
y_break_train = train_df[break_targets].astype("float32").values
y_break_val   =   val_df[break_targets].astype("float32").values

y_break_train_tensor = torch.tensor(y_break_train, dtype=torch.float32)
y_break_val_tensor   = torch.tensor(y_break_val,   dtype=torch.float32)
tau_train_tensor     = torch.tensor(tau_train,     dtype=torch.float32)
tau_val_tensor       = torch.tensor(tau_val,       dtype=torch.float32)

# ------------------ 1) Data Tensors (your primaries unchanged) ------------------
torch.manual_seed(42)

X_train_tensor = torch.tensor(X_train_proc.values, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_proc.values, dtype=torch.float32)
X_train_embed_tensor = torch.tensor(X_train_embed, dtype=torch.long)
X_val_embed_tensor   = torch.tensor(X_val_embed, dtype=torch.long)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val_scaled, dtype=torch.float32)

prior_train = torch.tensor(prior_train_scaled, dtype=torch.float32)
prior_val   = torch.tensor(prior_val_scaled, dtype=torch.float32)

# Add (y_break, τ) to the dataset; primary order unchanged
train_loader = DataLoader(
    TensorDataset(
        X_train_tensor, X_train_embed_tensor, y_train_tensor, prior_train,
        y_break_train_tensor, tau_train_tensor
    ),
    batch_size=512,
    shuffle=True
)

# ------------------ 2) Loss helpers ------------------
def soft_pinball_loss(preds, targets, tau=0.5, alpha=2.0):
    diff = targets - preds
    return torch.mean(F.softplus(alpha * (tau - (diff < 0).float()) * diff))

def kl_regularization(model, scale=1e-4):
    return sum((p ** 2).sum() for p in model.parameters()) * scale

def sharpness_penalty(lower, upper, target_width=5.0, scale=1e-3):
    w = (upper - lower).clamp(min=0.01)
    return scale * ((w - target_width) ** 2).mean()

def coverage_penalty(lower, upper, y_true, target=0.8, scale=1.0):
    covered = ((y_true >= lower) & (y_true <= upper)).float().mean()
    return scale * F.relu(target - covered)  # <-- flipped

# Breakout BCE: learns breakout probability, masks invalid rows, and balances per column
def aux_breakout_bce(aux_probs: torch.Tensor, y_raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(y_raw) & torch.isfinite(tau)
    labels = (y_raw > tau).float()
    valid_f = valid.float()
    pos = (labels * valid_f).sum(0) / valid_f.sum(0).clamp_min(1.0)
    pos = pos.clamp(1e-6, 1 - 1e-6)
    w_pos, w_neg = (1.0 - pos), pos
    weights = (labels * w_pos + (1.0 - labels) * w_neg) * valid_f
    eps = 1e-6
    loss = - (weights * (labels * torch.log(aux_probs.clamp_min(eps)) +
                         (1 - labels) * torch.log((1 - aux_probs).clamp_min(eps)))).sum(0)
    denom = weights.sum(0).clamp_min(1.0)
    return (loss / denom).mean()

# ------------------ 3) QuantileBNN (adds only aux heads, leave primaries intact) ------------------
class QuantileBNN(nn.Module):
    def __init__(self, input_dim, embedding_sizes, prior_dim, output_dim=6, dropout_rate=0.3, aux_dim: int = 0):
        super().__init__()
        self.output_dim = output_dim
        self.aux_dim = aux_dim

        self.embedding_dropout = nn.Dropout(0.1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in embedding_sizes
        ])
        emb_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        self.input_base_dim = input_dim + emb_total
        self.norm_input = nn.LayerNorm(self.input_base_dim)

        self.shared_base = nn.Sequential(
            nn.Linear(self.input_base_dim, 256), nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ELU(), nn.Dropout(dropout_rate)
        )

        # Primary heads (unchanged)
        self.heads_mean   = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_lower  = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_upper  = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_median = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_logvar = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])

        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(output_dim)])

        # New: one breakout-prob head per primary stat
        if self.aux_dim > 0:
            self.aux_logits = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.aux_dim)])

        # Temperature vector (unchanged)
        self.register_buffer("temp_vector", torch.tensor([5.0, 5.5, 5.1, 9.1, 6.9, 4.5], dtype=torch.float32))

    def _encode(self, x_num, x_embed):
        emb = [self.embedding_dropout(e(x_embed[:, i])) for i, e in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1)
        x = self.norm_input(x)
        return self.shared_base(x)

    def forward(self, x_num, x_embed, prior, use_prior=True):
        h = self._encode(x_num, x_embed)

        means, lowers, uppers, medians, logvars = [], [], [], [], []
        for i in range(self.output_dim):
            if use_prior and prior is not None:
                prior_mean = prior[:, i * 2 + 0].unsqueeze(1)
                a = torch.sigmoid(self.alpha[i])  # why: blend with prior early; learn a
                pred_mean = self.heads_mean[i](h)
                mean = a * pred_mean + (1 - a) * prior_mean
            else:
                mean = self.heads_mean[i](h)

            means.append(mean)
            lowers.append(self.heads_lower[i](h))
            uppers.append(self.heads_upper[i](h))
            medians.append(self.heads_median[i](h))
            logvars.append(self.heads_logvar[i](h))

        return (
            torch.cat(means, 1),
            torch.cat(lowers, 1),
            torch.cat(uppers, 1),
            torch.cat(medians, 1),
            torch.cat(logvars, 1),
        )

    # New: breakout probability heads (classification signal for 6 primaries)
    def forward_aux(self, x_num, x_embed):
        if self.aux_dim == 0:
            raise RuntimeError("No aux heads configured.")
        h = self._encode(x_num, x_embed)
        probs = [torch.sigmoid(hd(h)) for hd in self.aux_logits]
        return torch.cat(probs, dim=1)

# ------------------ 4) Primary loss (unchanged) ------------------
def total_loss(mean, lower, upper, median, logvar, y_true, model, weights,
               kl_scale=1e-4, sharp_scale=1e-3, coverage_scale=1.0):
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    var = torch.exp(logvar) + 1e-6
    hetero_loss = ((mean - y_true) ** 2 / var).mean() + var.mean()

    pinball_lower  = soft_pinball_loss(lower,  y_true, tau=0.1)
    pinball_upper  = soft_pinball_loss(upper,  y_true, tau=0.9)
    pinball_median = soft_pinball_loss(median, y_true, tau=0.5)

    kl    = kl_regularization(model, kl_scale)
    sharp = sharpness_penalty(lower, upper, scale=sharp_scale)
    cov   = coverage_penalty(lower, upper, y_true, scale=coverage_scale)

    return (weights * hetero_loss).mean() + pinball_lower + pinball_upper + pinball_median + kl + sharp + cov

# ------------------ 5) MC Dropout predict (unchanged) ------------------
def predict_mc(model, X_num, X_embed, prior_tensor, T=20):
    device = next(model.parameters()).device
    X_num, X_embed, prior_tensor = X_num.to(device), X_embed.to(device), prior_tensor.to(device)

    model.train()  # enable dropout for MC
    means, lowers, uppers, medians = [], [], [], []

    for _ in range(T):
        with torch.no_grad():
            m, lo, up, med, _ = model(X_num, X_embed, prior_tensor, use_prior=True)
            means.append(m)   # [B, 6]
            lowers.append(lo) # [B, 6]
            uppers.append(up) # [B, 6]
            medians.append(med)

    means   = torch.stack(means)     # [T, B, 6]
    lowers  = torch.stack(lowers)    # [T, B, 6]
    uppers  = torch.stack(uppers)    # [T, B, 6]
    medians = torch.stack(medians)   # [T, B, 6]

    mean_mc   = means.mean(0)
    std_mc    = means.std(0)         # diag-only uncertainty (for logging)
    lower_q   = lowers.mean(0)       # use learned quantile heads
    upper_q   = uppers.mean(0)
    median_q  = medians.mean(0)

    # Keep signature identical to your caller:
    return (mean_mc.cpu().numpy(),
            std_mc.cpu().numpy(),
            lower_q.cpu().numpy(),
            upper_q.cpu().numpy(),
            median_q.cpu().numpy())

# ------------------ 6) Training (add small aux loss; warmup) ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
prior_dim = prior_train.shape[1]

model = QuantileBNN(
    input_dim=X_train_tensor.shape[1],
    embedding_sizes=embedding_sizes,
    prior_dim=prior_dim,
    output_dim=y_train_tensor.shape[1],
    aux_dim=len(break_targets)  # exactly 6
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

target_weights = torch.tensor(
    [1.5 if t in ['points', 'rebounds', 'assists'] else 1.0 for t in targets],
    dtype=torch.float32
).to(device)

best_val_loss = float("inf")
patience = 3
epochs_no_improve = 0
aux_warmup = 3      # why: let primaries settle first
lambda_aux = 0.2    # small; avoid overpowering primaries

for epoch in range(100):
    model.train()
    train_loss = []

    for xb_num, xb_emb, yb, priorb, yb_break, taub in train_loader:
        xb_num, xb_emb, yb, priorb = xb_num.to(device), xb_emb.to(device), yb.to(device), priorb.to(device)
        yb_break, taub = yb_break.to(device), taub.to(device)

        use_prior = epoch >= aux_warmup

        mean, lower, upper, median, logvar = model(xb_num, xb_emb, priorb, use_prior=use_prior)
        loss_primary = total_loss(mean, lower, upper, median, logvar, yb, model, target_weights)

        # Add aux only after warmup
        if epoch >= aux_warmup:
            aux_probs = model.forward_aux(xb_num, xb_emb)
            loss_aux = aux_breakout_bce(aux_probs, yb_break, taub)
            loss = loss_primary + lambda_aux * loss_aux
        else:
            loss = loss_primary

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        mean, lower, upper, median, logvar = model(
            X_val_tensor.to(device),
            X_val_embed_tensor.to(device),
            prior_val.to(device),
            use_prior=True
        )
        val_loss = total_loss(mean, lower, upper, median, logvar, y_val_tensor.to(device), model, target_weights)

    print(f"Epoch {epoch+1} - Train Loss: {np.mean(train_loss):.4f} - Val Loss: {val_loss.item():.4f}")

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ------------------ 7) Load Best Model (unchanged) ------------------
model.load_state_dict(best_model_state)
model.eval()
model.to("cpu")
# ------------------ 8. Predict ------------------

y_val_mean, y_val_std, y_val_lower, y_val_upper, y_val_median = predict_mc(
    model,
    X_val_tensor.cpu(),
    X_val_embed_tensor.cpu(),
    prior_val.cpu(),
    T=50
)

# ------------------ 9. Inverse Transform ------------------

y_val_pred_unscaled   = y_scaler.inverse_transform(y_val_mean)
y_val_std_unscaled    = y_val_std * y_scaler.scale_
y_val_lower_unscaled  = y_scaler.inverse_transform(y_val_lower)
y_val_upper_unscaled  = y_scaler.inverse_transform(y_val_upper)
y_val_median_unscaled = y_scaler.inverse_transform(y_val_median)

# ------------------ 10. Evaluation ------------------

results = []
for i, target in enumerate(targets):
    y_true   = val_df[target].values
    y_pred   = y_val_pred_unscaled[:, i]
    y_lower  = y_val_lower_unscaled[:, i]
    y_upper  = y_val_upper_unscaled[:, i]
    y_median = y_val_median_unscaled[:, i]

    # Point estimate performance
    rmse_mean = root_mean_squared_error(y_true, y_pred)
    mae_mean  = mean_absolute_error(y_true, y_pred)
    r2        = r2_score(y_true, y_pred)

    # Median performance
    rmse_median = root_mean_squared_error(y_true, y_median)
    mae_median  = mean_absolute_error(y_true, y_median)

    # Pinball losses
    pinball_10 = mean_pinball_loss(y_true, y_lower, alpha=0.1)
    pinball_50 = mean_pinball_loss(y_true, y_median, alpha=0.5)
    pinball_90 = mean_pinball_loss(y_true, y_upper, alpha=0.9)

    # Coverage
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    results.append({
        "Target": target,
        "RMSE_Mean": rmse_mean,
        "MAE_Mean": mae_mean,
        "R2": r2,
        "RMSE_Median": rmse_median,
        "MAE_Median": mae_median,
        "Pinball_10": pinball_10,
        "Pinball_50": pinball_50,
        "Pinball_90": pinball_90,
        "80pct_Coverage": coverage
    })

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_parquet("datasets/Evaluation_Metrics.parquet")

# Aux Head Probabilities
with torch.no_grad():
    aux_val_probs = model.forward_aux(
        X_val_tensor.cpu(), X_val_embed_tensor.cpu()
    ).numpy()  # shape: [n_val, 6]
pd.DataFrame(aux_val_probs, columns=targets).to_parquet("datasets/AuxBreakout_Probs.parquet", index=False)

# ------------------ Create Directories ------------------
os.makedirs("models/bnn", exist_ok=True)
os.makedirs("models/clustering", exist_ok=True)
os.makedirs("pipelines", exist_ok=True)

# ------------------ Save Models ------------------
# Quantile BNN
torch.save(model, "models/bnn/nba_bnn_full_model.pt")
torch.save(model.state_dict(), "models/bnn/nba_bnn_weights_only.pt")

# Clustering model
joblib.dump(kmeans_final, "models/clustering/nba_player_clustering.joblib")

# ------------------ Save Pipelines ------------------
joblib.dump(preprocessor, "pipelines/preprocessor_pipeline.joblib")
joblib.dump(prior_pipeline, "pipelines/prior_pipeline.joblib")
joblib.dump(pca_pipeline, "pipelines/pca_pipeline.joblib")
joblib.dump(embed_encoder, "pipelines/embed_encoder.joblib")
joblib.dump(embedding_sizes, "pipelines/embedding_sizes.joblib")
joblib.dump(y_scaler, "pipelines/y_scaler.joblib")


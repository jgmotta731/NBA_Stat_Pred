"""
Created on Sun Apr 20 2025

@author: Jack Motta
"""

# ---------------------------------------------------
# Imports & Seed Setup
# ---------------------------------------------------
import os, random, joblib, gc, warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_curve, root_mean_squared_error, r2_score, precision_score, f1_score, recall_score, accuracy_score, silhouette_score, mean_absolute_error, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_pinball_loss, brier_score_loss, roc_curve
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import shap
import colorcet as cc
import re

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
pd.set_option('display.max_columns', 20)

# ---------------------------------------------------
# Load & Prepare Data
# ---------------------------------------------------
# --- Normalize names ---
def normalize_name(name):
    name = name.strip()
    # Flip name if it's in "Last, First" format
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    # Then clean punctuation and Roman numerals
    name = re.sub(r'[^\w\s]', '', name)              # remove punctuation
    name = re.sub(r'\b(?:[IVX]+)$', '', name)        # remove trailing Roman numerals
    return name.strip()

# Load the 2025 injury data
injuries_2025 = pd.read_csv('2025_injuries.csv')

# Load the historical injury database
injury_db = pd.read_csv('Injury Database.csv')

# Vertically concatenate the two datasets (rows from 2025 added to historical)
injury_db = pd.concat([injury_db, injuries_2025], ignore_index=True)

gamelogs = pd.read_parquet('nba_gamelogs.parquet')
# downcast numeric types
gamelogs[gamelogs.select_dtypes('float64').columns] = \
    gamelogs.select_dtypes('float64')\
            .apply(pd.to_numeric, downcast='float')
gamelogs[gamelogs.select_dtypes('int64').columns] = \
    gamelogs.select_dtypes('int64')\
            .apply(pd.to_numeric, downcast='integer')
        
injury_db['norm'] = injury_db['PLAYER'].apply(normalize_name)
gamelogs['norm'] = (
    gamelogs['athlete_display_name']
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\b(?:[IVX]+)$', '', regex=True)
        .str.strip()
)

# --- Standardize dates ---
injury_db['DATE'] = pd.to_datetime(injury_db['DATE'], errors='coerce')
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'], errors='coerce')

# --- Filter to 2020-2021 season onward ---
injury_db = injury_db[injury_db['DATE'].dt.year >= 2020].copy()
gamelogs = gamelogs[gamelogs['season'] >= 2021].copy().reset_index(drop=True)

# --- Identify starters (avg minutes >= 30) ---
starter_threshold = 30
starter_avg = gamelogs.groupby('norm')['minutes'].mean()
starters = set(starter_avg[starter_avg >= starter_threshold].index)

# Filter only starter injuries
starter_injuries = injury_db[injury_db['norm'].isin(starters)].copy()

# Merge with gamelogs on team and date
starter_injuries['starter_injured'] = True
gamelogs = gamelogs.merge(
    starter_injuries[['TEAM', 'DATE', 'starter_injured']],
    left_on=['team_display_name', 'game_date'],
    right_on=['TEAM', 'DATE'],
    how='left'
)

# Step 3: Fill missing with False
gamelogs['starter_injured'] = gamelogs['starter_injured'].fillna(False)

# Drop merge columns
gamelogs = gamelogs.drop(columns=['TEAM', 'DATE', 'team_display_name']).reset_index(drop=True)

# Response variables
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']

# Drop games with missing target stats (i.e., not played)
gamelogs = gamelogs.dropna(subset=targets).copy()

# Count games per player
player_game_counts = gamelogs.groupby("athlete_display_name").size()

# Keep only players with >=10 valid games
valid_players = player_game_counts[player_game_counts >= 20].index
gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

# After filtering players ➔ filter low-minute games
gamelogs = gamelogs[gamelogs["did_not_play"] == False].reset_index(drop=True)

# Get Game Odds
hoopr_odds = pd.read_parquet('hoopr_game_odds.parquet')

# Ensure team_abbreviation and opponent_team_abbreviation are uppercase
gamelogs['team_abbreviation'] = gamelogs['team_abbreviation'].str.upper()
gamelogs['opponent_team_abbreviation'] = gamelogs['opponent_team_abbreviation'].str.upper()

# Make home and away columns based on home_away flag
gamelogs['home'] = gamelogs.apply(
    lambda row: row['team_abbreviation'] if row['home_away'].lower() == 'home' else row['opponent_team_abbreviation'],
    axis=1
)

gamelogs['away'] = gamelogs.apply(
    lambda row: row['team_abbreviation'] if row['home_away'].lower() == 'away' else row['opponent_team_abbreviation'],
    axis=1
)

# Ensure date columns are datetime for join
hoopr_odds['game_date'] = pd.to_datetime(hoopr_odds['game_date'])

# Ensure uppercase consistency
hoopr_odds['home_team'] = hoopr_odds['home_team'].str.upper()
hoopr_odds['away_team'] = hoopr_odds['away_team'].str.upper()

# Ensure date formats are aligned
hoopr_odds['game_date'] = pd.to_datetime(hoopr_odds['game_date'])

# Left join on HoopR odds data from play-by-play data
gamelogs = gamelogs.merge(
    hoopr_odds,
    how='left',
    left_on=['game_date', 'home', 'away', 'season'],
    right_on=['game_date', 'home_team', 'away_team', 'season'],
    suffixes=('', '_hoopr')  # optional: avoid column collisions
)

# Drop temporary columns
gamelogs = gamelogs.drop(columns=['game_id_hoopr', 'home_team', 'away_team', 'home', 'away', 'norm'])

# Downcast float64 to float32
gamelogs['game_spread'] = gamelogs['game_spread'].astype(np.float32)
gamelogs['home_team_spread'] = gamelogs['home_team_spread'].astype(np.float32)

# Clean up
del hoopr_odds, injury_db, injuries_2025, starter_avg, starter_injuries, starter_threshold, starters, valid_players, player_game_counts
gc.collect()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
# Ensure game_date is datetime
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])

# Drop duplicates
gamelogs = gamelogs.drop_duplicates().reset_index(drop=True)

# Global temporal sort
gamelogs = gamelogs.sort_values(['athlete_display_name', 'game_date', 'season', 'team_abbreviation']).reset_index(drop=True)

# Clean plus_minus
if 'plus_minus' in gamelogs.columns:
    gamelogs['plus_minus'] = (
        gamelogs['plus_minus']
        .astype(str)
        .str.replace(r'^\+', '', regex=True)
        .replace('None', np.nan)
        .astype(np.float32)
    )

# Booleans to numeric
bool_cols = ['starter', 'ejected', 'team_winner', 'is_playoff', 'starter_injured', 'home_favorite']
for col in bool_cols:
    if col in gamelogs.columns:
        gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)

# Playoff indicator
gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)

# Days since last game
gamelogs['days_since_last_game'] = gamelogs.groupby(['athlete_display_name', 'season'])['game_date'].diff().dt.days.astype(np.float32)

# Back-to-back
gamelogs['is_back_to_back'] = (gamelogs['days_since_last_game'].fillna(99) == 1).astype(np.int32)

# Rolling columns
rolling_cols = ['field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
                'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
                'rebounds', 'assists', 'steals', 'blocks', 'points', 'minutes']

def compute_lag_features(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    result[f'{col}_lag1'] = df.groupby(['athlete_display_name', 'season'])[col].shift(1)
    result[f'{col}_lag2'] = df.groupby(['athlete_display_name', 'season'])[col].shift(2)
    result[f'{col}_lag3'] = df.groupby(['athlete_display_name', 'season'])[col].shift(3)
    return result

lag_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_lag_features)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + lag_results, axis=1)

def compute_expanding_mean(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    result[f'{col}_expanding_mean'] = (
        df.groupby(['athlete_display_name', 'season'])[col]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    return result

# Apply in parallel
expanding_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_expanding_mean)(gamelogs, col) for col in rolling_cols
)

# Concatenate to main DataFrame
gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)

def compute_rolling_and_ewm_features(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    ewm_spans = [5, 9, 19]
    for span in ewm_spans:
        shifted = df.groupby(['athlete_display_name', 'season'])[col].shift(1)
        shift2 = df.groupby(['athlete_display_name', 'season'])[col].shift(2)
        ewm_mean = shifted.groupby([df['athlete_display_name'], df['season']]) \
                          .ewm(span=span, adjust=False).mean().reset_index(level=[0,1], drop=True)
        ewm_shift2_mean = shift2.groupby([df['athlete_display_name'], df['season']]) \
                                .ewm(span=span, adjust=False).mean().reset_index(level=[0,1], drop=True)
        result[f'{col}_ewm_mean_span{span}'] = ewm_mean
        result[f'{col}_ewm_momentum_span{span}'] = shifted - ewm_shift2_mean
        result[f'{col}_ewm_zscore_span{span}'] = (shifted - ewm_mean) / (ewm_mean.std() + 1e-6)
        temp_ewm = df[['game_date']].copy()
        temp_ewm[f'{col}_ewm_mean_span{span}'] = ewm_mean
        result[f'{col}_ewm_mean_span{span}_rank'] = temp_ewm.groupby('game_date')[f'{col}_ewm_mean_span{span}'].transform(lambda x: x.rank(pct=True)).astype(np.float32)
    return result

rolling_ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_rolling_and_ewm_features)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + rolling_ewm_results, axis=1)

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
                weights = pd.Series(np.arange(1, len(y) + 1)).ewm(span=span, adjust=False).mean().to_numpy()
                weights = weights[mask]
                x_masked = x[mask]
                y_masked = y[mask]
                model = LinearRegression()
                model.fit(x_masked, y_masked, sample_weight=weights)
                slopes[i] = model.coef_[0]
            except:
                pass
    return idx, slopes

for span in [5, 9, 19]:
    for col in targets:
        groups = gamelogs.groupby(['athlete_display_name', 'season'])
        results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
            delayed(compute_ewm_trend_stats_for_player)(name, grp, col, span)
            for name, grp in groups)
        slope_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)
        for idxs, slopes in results:
            slope_arr[idxs] = slopes
        gamelogs[f'{col}_ewm_trend_slope_{span}'] = slope_arr

# EWM STD
gamelogs['points_ewm_std_span9'] = gamelogs.groupby(['athlete_display_name', 'season'])['points'].transform(lambda x: x.shift(1).ewm(span=9, adjust=False).std())

# Hot/cold flags
gamelogs['is_hot_game'] = ((gamelogs['points'] >= gamelogs['points_ewm_mean_span9'] + gamelogs['points_ewm_std_span9']).fillna(False).astype(int))
gamelogs['is_cold_game'] = ((gamelogs['points'] < gamelogs['points_ewm_mean_span9'] - gamelogs['points_ewm_std_span9']).fillna(False).astype(int))

# Streaks
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
gamelogs = gamelogs.drop(columns=[
    'is_hot_game', 'is_cold_game', 'points_ewm_std_span9'
], axis=1)

# Opponent stats to compute
opponent_stats = ['points', 'rebounds', 'assists', 'three_point_field_goals_made']
ewm_spans = [5, 9, 19]

# Compute opponent-based EWM allowed stats
def compute_opponent_ewm_allowed(stat, span):
    # Step 1: Aggregate total stat allowed by opponent per game
    opponent_daily = (
        gamelogs
        .groupby(['opponent_team_abbreviation', 'season', 'game_date'])[stat]
        .sum()
        .reset_index()
        .sort_values(['opponent_team_abbreviation', 'season', 'game_date'])
    )

    # Step 2: Apply shifted EWM to avoid leakage
    ewm_col = f'opponent_ewm_{stat}_allowed_span{span}'
    opponent_daily[ewm_col] = (
        opponent_daily
        .groupby(['opponent_team_abbreviation', 'season'])[stat]
        .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    )

    return opponent_daily[['opponent_team_abbreviation', 'season', 'game_date', ewm_col]]

# Run in parallel for all stat/span combinations
opponent_ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_opponent_ewm_allowed)(stat, span)
    for stat in opponent_stats
    for span in ewm_spans
)

# Merge each result back into gamelogs
for result in opponent_ewm_results:
    gamelogs = gamelogs.merge(result, on=['opponent_team_abbreviation', 'season', 'game_date'], how='left')

# Specify Interactions
for span in [5, 9, 19]:
    gamelogs[f'interaction_points_ewm_span{span}'] = (
        gamelogs[f'points_ewm_mean_span{span}'] *
        gamelogs[f'opponent_ewm_points_allowed_span{span}']
    )
    gamelogs[f'interaction_rebounds_ewm_span{span}'] = (
        gamelogs[f'rebounds_ewm_mean_span{span}'] *
        gamelogs[f'opponent_ewm_rebounds_allowed_span{span}']
    )
    gamelogs[f'interaction_assists_ewm_span{span}'] = (
        gamelogs[f'assists_ewm_mean_span{span}'] *
        gamelogs[f'opponent_ewm_assists_allowed_span{span}']
    )
    gamelogs[f'interaction_3pm_ewm_span{span}'] = (
        gamelogs[f'three_point_field_goals_made_ewm_mean_span{span}'] *
        gamelogs[f'opponent_ewm_three_point_field_goals_made_allowed_span{span}']
    )

# Drop Opponent Columns, Leaving Interactions Only
gamelogs = gamelogs.drop(columns=[col for col in gamelogs.columns if col.startswith("opponent_ewm_")])

# Downcast float64 to float32
float_cols = gamelogs.select_dtypes(include=['float64']).columns
gamelogs[float_cols] = gamelogs[float_cols].astype(np.float32)

# Downcast int64 to int32
int_cols = gamelogs.select_dtypes(include=['int64']).columns
gamelogs[int_cols] = gamelogs[int_cols].astype(np.int32)

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------
gamelogs = gamelogs[gamelogs["season"] >= 2022].copy().reset_index(drop=True)

gamelogs = gamelogs.drop(columns=[
    'plus_minus', 'ejected', 'did_not_play', 'team_winner', 'active',
    'season_type', 'starter'
    ])

lagged_rolling_features = [col for col in gamelogs.columns
                           if 'rolling' in col or 'trend' in col or 'lag' in col 
                           or 'expanding' in col or 'momentum' in col or 'streak' in col
                           or 'span' in col or 'ewm' in col or 'spread' in col]

numeric_features = lagged_rolling_features + ['days_since_last_game']
categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff",
                        "starter_injured", "is_back_to_back", "home_favorite"]
features = numeric_features + categorical_features

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

# ---------------------------------------------------
# PCA + Clustering (with Temporal Train/Test Split)
# ---------------------------------------------------
# Primary + Secondary Targets
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made', 
            'free_throws_attempted', 'free_throws_made',
            'three_point_field_goals_attempted'] + targets

# Define EWM feature columns
ewm_stat_cols = [f"{target}_ewm_mean_span9" for target in targets2]

# All clustering features (EWM + shifted cumulative flags)
cluster_features = ewm_stat_cols

# Temporal Split
train_gamelogs = gamelogs[gamelogs['season'] < 2025].copy()

# Aggregate to player-level for clustering
train_player_summary = train_gamelogs.groupby('athlete_display_name')[cluster_features].mean().reset_index()

# Preprocessing pipeline: median impute + scale
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), cluster_features)
])

# Fit PCA only on training player summary
X_train_proc = preprocessor.fit_transform(train_player_summary[cluster_features])
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
plt.bar(components[:100], explained[:100], width=0.8)
plt.xlim(-2, 100)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot (First 100 Components)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Fit PCA pipeline (just PC1)
pca_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=1, random_state=SEED))
])

X_cluster_train = pca_pipeline.fit_transform(train_player_summary[cluster_features])

# Extract PCA loadings
pca_model = pca_pipeline.named_steps['pca']
loadings = pca_model.components_
feature_names = pca_pipeline.named_steps['preprocessor'].get_feature_names_out(cluster_features)

# Top contributors to PC1
num_pcs_to_show = min(20, loadings.shape[0])
for i in range(num_pcs_to_show):
    pc_loadings = pd.Series(loadings[i], index=feature_names, name=f'PC{i+1}_Loading') \
                    .sort_values(key=abs, ascending=False)
    print(f"\nTop contributors to PC{i+1}:")
    print(pc_loadings)

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

# Final KMeans
kmeans_final = KMeans(n_clusters=6, random_state=SEED)
kmeans_final.fit(X_cluster_train)

# Assign clusters
train_player_summary['cluster_label'] = kmeans_final.labels_ + 1

# Group by cluster and compute means
cluster_means = (
    train_player_summary
    .groupby('cluster_label')[cluster_features]
    .mean()
    .round(2)  # Optional: round for cleaner display
    .sort_index()
)

# Display
print(cluster_means)

# Assume you have PC1 and cluster_label
pca_1d_df = pd.DataFrame(X_cluster_train, columns=['PC1'])
pca_1d_df['cluster_label'] = train_player_summary['cluster_label'].values

# Use the first 26 Glasbey colors
distinct_palette = cc.glasbey[:25]
plt.figure(figsize=(12, 8))
sns.stripplot(data=pca_1d_df, x='PC1', hue='cluster_label', 
              palette=distinct_palette,
              jitter=0.1, size=8, alpha=0.8)
plt.title('1D PCA Cluster Distribution (PC1)', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.yticks([])
plt.legend(title='Cluster', bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Add Cluster Labels to Gamelogs
# ---------------------------------------------------
# Build a full player summary (using only features trained above)
full_player_summary = (
    gamelogs
    .groupby('athlete_display_name')[numeric_features]
    .mean()
    .reset_index()
)

# Transform with the pretrained pipeline and predict cluster labels
X_full_cluster = pca_pipeline.transform(full_player_summary[numeric_features])
full_player_summary['cluster_label'] = kmeans_final.predict(X_full_cluster) + 1

# Extract just name + cluster_label
player_clusters = full_player_summary[['athlete_display_name', 'cluster_label']].copy()
player_clusters['cluster_label'] = player_clusters['cluster_label'].astype('float32')
player_clusters.to_parquet('player_clusters.parquet')

# Merge cluster labels into gamelogs before any train/val split
gamelogs = (
    gamelogs
    .drop(columns=['cluster_label'], errors='ignore')
    .merge(player_clusters, on='athlete_display_name', how='left')
)

# Now update your feature lists
features += ['cluster_label', 'was_missing']
categorical_features += ['cluster_label', 'was_missing']

# Convert int32 columns to float32 for consistency
gamelogs[gamelogs.select_dtypes('int32').columns] = (
    gamelogs.select_dtypes('int32').astype('float32')
)

# ---------------------------------------------------
# Train/Validation Split
# ---------------------------------------------------
# Ensure uniqueness in lists
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))
features = list(dict.fromkeys(features))

train_df = gamelogs[gamelogs["season"] < 2025].copy()
val_df = gamelogs[gamelogs["season"] >= 2025].copy()

explosive_thresholds = {
    'points': 30,
    'rebounds': 10,
    'assists': 10,
    'three_point_field_goals_made': 5,
    'steals': 3,
    'blocks': 3
}

for stat, threshold in explosive_thresholds.items():
    train_df[f'explosive_{stat}'] = (train_df[stat] >= threshold).astype(int)
    val_df[f'explosive_{stat}']   = (val_df[stat] >= threshold).astype(int)


# Force categorical columns to be string type for compatibility
for col in categorical_features:
    train_df[col] = train_df[col].astype(str)
    val_df[col] = val_df[col].astype(str)
    
# ---------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------
X_train = train_df[features]
X_val = val_df[features]

# Split targets
y_train_regression = train_df[targets]
y_val_regression = val_df[targets]

# Preprocessing pipeline
column_transformer = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32))
    ]), categorical_features)
])

# Wrap ColumnTransformer inside a full pipeline
preprocessor = Pipeline([
    ('transform', column_transformer),
    ('variance', VarianceThreshold(threshold=0.05))
])

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

# ---------------------------------------------------
# All Targets Model
# ---------------------------------------------------
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made', 
            'free_throws_attempted', 'free_throws_made',
            'three_point_field_goals_attempted'] + targets

y_train_regression2 = train_df[targets2]
y_val_regression2 = val_df[targets2]

# Ridge Regression base learner
ridge_base = Ridge(alpha=5.0, random_state=42)
ridge_mt   = MultiOutputRegressor(ridge_base, n_jobs=-1)
ridge_mt.fit(X_train_proc, y_train_regression2)

# Predict on train/val
y_train_pred_ridge = ridge_mt.predict(X_train_proc)
y_val_pred_ridge   = ridge_mt.predict(X_val_proc)

# Evaluate Ridge predictions
metrics_list = []

for idx, target in enumerate(targets2):
    y_true = y_val_regression2.iloc[:, idx]
    y_pred = y_val_pred_ridge[:, idx]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics_list.append({
        'target': target,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

metrics_df = pd.DataFrame(metrics_list)
print("\n📊 Ridge Regression (Base Learner) Evaluation:")
print(metrics_df)

# Feature names after preprocessing
num_feat = column_transformer.named_transformers_['num'].named_steps['scale'].get_feature_names_out(numeric_features)
cat_feat = column_transformer.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_features)
all_features = np.concatenate([num_feat, cat_feat])

# Apply VarianceThreshold mask
mask = preprocessor.named_steps['variance'].get_support()
feature_names = all_features[mask]

# Collect Ridge coefficients per target
coefs_ridge = []
for i, estimator in enumerate(ridge_mt.estimators_):
    coef = estimator.coef_.flatten()
    coefs_ridge.append(pd.Series(coef, index=feature_names, name=targets2[i]))

# Combine and compute mean absolute coefficient
coef_df_ridge = pd.concat(coefs_ridge, axis=1)
coef_df_ridge['mean_abs_coef'] = coef_df_ridge.abs().mean(axis=1)
coef_df_ridge_sorted = coef_df_ridge.sort_values('mean_abs_coef', ascending=False)

# Plot top N features
TOP_N = 20
top_features_ridge = coef_df_ridge_sorted.head(TOP_N)

plt.figure(figsize=(10, 6))
top_features_ridge['mean_abs_coef'].plot(kind='barh', color='teal')
plt.gca().invert_yaxis()
plt.title(f'Top {TOP_N} Features (Ridge Regression Avg Coef Magnitude)')
plt.xlabel('Mean Absolute Coefficient')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Bin Classification Model
# ---------------------------------------------------
bin_edges = {
    'minutes': [0, 29, np.inf],
    'field_goals_attempted': [0, 10, np.inf],
    'field_goals_made': [0, 5, np.inf],
    'free_throws_attempted': [0, 2, np.inf],
    'free_throws_made': [0, 2, np.inf],
    'three_point_field_goals_attempted': [0, 4, np.inf],
    'points': [0, 14, np.inf],
    'assists': [0, 3, np.inf],
    'rebounds': [0, 5, np.inf],
    'steals': [0, 1, np.inf],
    'blocks': [0, 1, np.inf],
    'three_point_field_goals_made': [0, 2, np.inf],
}

def create_bins(df, targets, bin_edges):
    binned_labels = {}
    for target in targets:
        bins = bin_edges[target]
        binned_labels[target] = pd.cut(
            df[target],
            bins=bins,
            labels=False,
            include_lowest=True,
            right=False
        )
    return pd.DataFrame(binned_labels, index=df.index)

y_train_bins = create_bins(train_df, targets2, bin_edges)
y_val_bins = create_bins(val_df, targets2, bin_edges)

# Check class distribution for each target
class_distributions = {}

for target in y_train_bins.columns:
    counts = y_train_bins[target].value_counts().sort_index()
    class_distributions[target] = counts

# Turn into a readable table
distribution_df = pd.DataFrame(class_distributions).fillna(0).astype(int)
distribution_df.index.name = 'Bin'

print("\nClass distributions per target:")
print(distribution_df)

# Base logistic model
logreg_clf = LogisticRegression(
    penalty='l1',
    C = 5.0,
    solver='saga',
    max_iter=10000,
    random_state=SEED,
    n_jobs=-1
)

# Multi-output wrapper
multi_logreg = MultiOutputClassifier(logreg_clf, n_jobs=-1)
multi_logreg.fit(X_train_proc, y_train_bins)

# Predict on train and val
y_train_pred_lr = multi_logreg.predict(X_train_proc)
y_val_pred_lr   = multi_logreg.predict(X_val_proc)

# Evaluate
metrics_list = []
for idx, target in enumerate(y_val_bins.columns):
    y_true = y_val_bins.iloc[:, idx]
    y_pred = y_val_pred_lr[:, idx]

    # Probabilities for class 1
    y_prob = multi_logreg.estimators_[idx].predict_proba(X_val_proc)[:, 1]

    metrics_list.append({
        'target': target,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob)
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# Feature Importance
# Collect coefficients for each target
coefs = []
for i, estimator in enumerate(multi_logreg.estimators_):
    coef = estimator.coef_.flatten()  # LogisticRegression is binary so shape = (1, n_features)
    coefs.append(pd.Series(coef, index=feature_names, name=y_train_bins.columns[i]))

# Combine into one DataFrame
coef_df = pd.concat(coefs, axis=1)
coef_df['mean_abs_coef'] = coef_df.abs().mean(axis=1)
coef_df_sorted = coef_df.sort_values('mean_abs_coef', ascending=False)

# Plot top N features
TOP_N = 20
top_features = coef_df_sorted.head(TOP_N)

plt.figure(figsize=(10, 6))
top_features['mean_abs_coef'].plot(kind='barh')
plt.gca().invert_yaxis()
plt.title(f'Top {TOP_N} Most Important Features (LogReg Avg Coef Magnitude)')
plt.xlabel('Mean Absolute Coefficient')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Calibration Curves
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for idx, target in enumerate(y_val_bins.columns[:6]):
    ax = axs[idx // 3, idx % 3]
    y_true = y_val_bins.iloc[:, idx]
    y_prob = multi_logreg.estimators_[idx].predict_proba(X_val_proc)[:, 1]

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax.set_title(f'Calibration Curve: {target}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Empirical Probability')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()

# Calibrate each base logistic regression model
calibrated_estimators = []

for i, (target, base_clf) in enumerate(zip(y_train_bins.columns, multi_logreg.estimators_)):
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method='isotonic',
        cv='prefit', n_jobs=-1
    )
    calibrated.fit(X_train_proc, y_train_bins.iloc[:, i])
    calibrated_estimators.append(calibrated)

# Use the calibrated models, not the original multi_logreg
y_train_pred_lr_proba = np.column_stack([
    clf.predict_proba(X_train_proc)[:, 1]
    for clf in calibrated_estimators
])

y_val_pred_lr_proba = np.column_stack([
    clf.predict_proba(X_val_proc)[:, 1]
    for clf in calibrated_estimators
])

# Predicted class labels (bins) for training and validation
y_train_pred_lr = np.column_stack([
    clf.predict(X_train_proc) for clf in calibrated_estimators
])

y_val_pred_lr = np.column_stack([
    clf.predict(X_val_proc) for clf in calibrated_estimators
])

metrics = []
for i, (target, model) in enumerate(zip(y_val_bins.columns, calibrated_estimators)):
    y_true = y_val_bins.iloc[:, i]
    y_pred = model.predict(X_val_proc)
    y_prob = model.predict_proba(X_val_proc)[:, 1]

    metrics.append({
        'target': target,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob)
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Calibration Curves of Calibrated Models
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
for idx, target in enumerate(y_val_bins.columns[:6]):
    ax = axs[idx // 3, idx % 3]
    y_true = y_val_bins.iloc[:, idx]
    
    # Use calibrated model probabilities
    y_prob = calibrated_estimators[idx].predict_proba(X_val_proc)[:, 1]
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', label='Calibrated')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_title(f'Calibration Curve (Calibrated): {target}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Empirical Probability')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for idx, (target, model) in enumerate(zip(y_val_bins.columns[:6], calibrated_estimators[:6])):
    y_true = y_val_bins[target]
    y_prob = model.predict_proba(X_val_proc)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"{target} (AUC = {auc:.3f})")

# Random chance line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")

# Plot formatting
plt.title("ROC Curves (Calibrated Logistic Models)", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
plt.close('all')
# ---------------------------------------------------
# Explosive Game Classification
# ---------------------------------------------------
for stat, threshold in explosive_thresholds.items():
    col = f'explosive_{stat}'
    train_df[col] = (train_df[stat] >= threshold).astype(int)
    val_df[col] = (val_df[stat] >= threshold).astype(int)
    print(f"Train class balance for {col}: {Counter(train_df[col])}")
    print(f"Val class balance   for {col}: {Counter(val_df[col])}\n")

# Oversample targets
ros = RandomOverSampler(random_state=SEED, sampling_strategy=0.5)

# Shared 2x3 calibration subplot setup
fig_cal, axs_cal = plt.subplots(2, 3, figsize=(18, 10))

# Loop each target through model fitting, feature imp plot, calibration, and cal curve plots
explosive_results = {}
calibration_data = {}
uncalibrated_metrics_list = []
calibrated_metrics_list = []

for stat in explosive_thresholds.keys():
    print(f"\n--- Explosive Classification for {stat.title()} ---")
    target_col = f"explosive_{stat}"

    # Resample
    X_train_bal, y_train_bal = ros.fit_resample(X_train_proc, train_df[target_col])

    # Fit base model
    model = LogisticRegression(
        solver='saga',
        max_iter=10000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED,
        penalty='l1',
        C=5.0
    )
    model.fit(X_train_bal, y_train_bal)

    # Predict (Uncalibrated)
    y_val_true = val_df[target_col]
    y_val_pred = model.predict(X_val_proc)
    y_val_prob = model.predict_proba(X_val_proc)[:, 1]

    # Uncalibrated metrics (threshold = 0.5)
    uncal_metrics = {
        "target": target_col,
        "model": type(model).__name__,
        "accuracy": accuracy_score(y_val_true, y_val_pred),
        "precision": precision_score(y_val_true, y_val_pred, zero_division=0),
        "recall": recall_score(y_val_true, y_val_pred, zero_division=0),
        "f1": f1_score(y_val_true, y_val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val_true, y_val_prob),
        "brier_score": brier_score_loss(y_val_true, y_val_prob),
        "confusion_matrix": confusion_matrix(y_val_true, y_val_pred).tolist()
    }
    uncalibrated_metrics_list.append(uncal_metrics)

    # Absolute Coefficient Importance
    coef_vals = model.coef_.flatten()
    importance_df = pd.Series(np.abs(coef_vals), index=feature_names).sort_values(ascending=False)

    # Plot top N most important features
    TOP_N = 20
    fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
    importance_df.head(TOP_N).plot(kind="barh", color="darkorange", ax=ax_feat)
    ax_feat.invert_yaxis()
    ax_feat.set_title(f"Top {TOP_N} Features (|Coefficient|) - Explosive {stat.title()}")
    ax_feat.set_xlabel("Mean Absolute Coefficient")
    ax_feat.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Calibrate model using isotonic (no refit)
    calibrated_model = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_train_proc, train_df[target_col])
    y_val_prob_cal = calibrated_model.predict_proba(X_val_proc)[:, 1]

    # Save for calibration curve
    calibration_data[stat] = {
        "y_val_true": y_val_true,
        "y_val_prob_uncal": y_val_prob,
        "y_val_prob_cal": y_val_prob_cal
    }

    # Plot calibration curve (assuming axs_cal was created beforehand)
    idx = list(explosive_thresholds.keys()).index(stat)
    row, col = divmod(idx, 3)
    ax = axs_cal[row, col]
    prob_true, prob_pred = calibration_curve(y_val_true, y_val_prob_cal, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', label='Calibrated')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title(f'Calibration Curve: {stat}')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Empirical Probability")
    ax.grid(True)

    # Find best threshold using F1
    precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob_cal)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = f1_scores.argmax()
    best_thresh = thresholds[best_idx]

    # Calibrated metrics (threshold = best_thresh)
    y_val_pred_cal = (y_val_prob_cal >= best_thresh).astype(int)
    cal_metrics = {
        "target": target_col,
        "model": "Calibrated_" + type(model).__name__,
        "accuracy": accuracy_score(y_val_true, y_val_pred_cal),
        "precision": precision_score(y_val_true, y_val_pred_cal, zero_division=0),
        "recall": recall_score(y_val_true, y_val_pred_cal, zero_division=0),
        "f1": f1_score(y_val_true, y_val_pred_cal, zero_division=0),
        "roc_auc": roc_auc_score(y_val_true, y_val_prob_cal),
        "brier_score": brier_score_loss(y_val_true, y_val_prob_cal),
        "confusion_matrix": confusion_matrix(y_val_true, y_val_pred_cal).tolist(),
        "best_thresh": best_thresh
    }
    calibrated_metrics_list.append(cal_metrics)

    # Final predictions
    train_proba = calibrated_model.predict_proba(X_train_proc)[:, 1]
    val_proba = y_val_prob_cal
    train_pred = (train_proba >= best_thresh).astype(int)
    val_pred = (val_proba >= best_thresh).astype(int)

    # Store everything
    explosive_results[f"{stat}_calibrated_model"] = calibrated_model
    explosive_results[f"{stat}_metrics_uncal"] = uncal_metrics
    explosive_results[f"{stat}_metrics_cal"] = cal_metrics
    explosive_results[f"{stat}_importance"] = importance_df
    explosive_results[f"{stat}_train_proba"] = train_proba
    explosive_results[f"{stat}_val_proba"] = val_proba
    explosive_results[f"{stat}_train_pred"] = train_pred
    explosive_results[f"{stat}_val_pred"] = val_pred
    explosive_results[f"{stat}_best_thresh"] = best_thresh

# After loop: combine and print full tables
uncalibrated_df = pd.DataFrame(uncalibrated_metrics_list).set_index("target")
calibrated_df = pd.DataFrame(calibrated_metrics_list).set_index("target")

print("\n=== Uncalibrated Metrics Summary ===")
print(uncalibrated_df)

print("\n=== Calibrated Metrics Summary ===")
print(calibrated_df)

# Stack predictions and probabilities
explosive_train_preds = np.column_stack([
    explosive_results[f"{stat}_train_pred"] for stat in explosive_thresholds.keys()
])

explosive_val_preds = np.column_stack([
    explosive_results[f"{stat}_val_pred"] for stat in explosive_thresholds.keys()
])

explosive_train_probas = np.column_stack([
    explosive_results[f"{stat}_train_proba"] for stat in explosive_thresholds.keys()
])

explosive_val_probas = np.column_stack([
    explosive_results[f"{stat}_val_proba"] for stat in explosive_thresholds.keys()
])

# Uncalibrated Calibration Curves
fig_uncal, axs_uncal = plt.subplots(2, 3, figsize=(18, 10))
fig_uncal.suptitle("Calibration Curves (Before Calibration)", fontsize=16)

# Calibrated Calibration Curves
fig_cal, axs_cal = plt.subplots(2, 3, figsize=(18, 10))
fig_cal.suptitle("Calibration Curves (After Calibration)", fontsize=16)
for idx, (stat, data) in enumerate(calibration_data.items()):
    row, col = divmod(idx, 3)

    # Uncalibrated
    ax_uncal = axs_uncal[row, col]
    prob_true_uncal, prob_pred_uncal = calibration_curve(data["y_val_true"], data["y_val_prob_uncal"], n_bins=10)
    ax_uncal.plot(prob_pred_uncal, prob_true_uncal, linestyle="--", marker="o", label="Uncalibrated")
    ax_uncal.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax_uncal.set_title(f"{stat.title()}")
    ax_uncal.set_xlabel("Predicted Probability")
    ax_uncal.set_ylabel("True Probability")
    ax_uncal.grid(True, linestyle="--", alpha=0.5)
    ax_uncal.legend()

    # Calibrated
    ax_cal = axs_cal[row, col]
    prob_true_cal, prob_pred_cal = calibration_curve(data["y_val_true"], data["y_val_prob_cal"], n_bins=10)
    ax_cal.plot(prob_pred_cal, prob_true_cal, linestyle="-", marker="o", label="Calibrated", color="darkorange")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax_cal.set_title(f"{stat.title()}")
    ax_cal.set_xlabel("Predicted Probability")
    ax_cal.set_ylabel("True Probability")
    ax_cal.grid(True, linestyle="--", alpha=0.5)
    ax_cal.legend()
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for stat, data in calibration_data.items():
    y_true = data["y_val_true"]
    y_prob_cal = data["y_val_prob_cal"]
    fpr, tpr, _ = roc_curve(y_true, y_prob_cal)
    auc = roc_auc_score(y_true, y_prob_cal)
    plt.plot(fpr, tpr, label=f"{stat.title()} (AUC = {auc:.3f})")

# Plot baseline
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
plt.title("ROC Curves for Calibrated Explosive Classifiers", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Stack Predictions
# ---------------------------------------------------
X_train_stacked = np.hstack([
    X_train_proc,
    y_train_pred_lr,
    y_train_pred_lr_proba,
    explosive_train_preds,
    explosive_train_probas,
    y_train_pred_ridge
])

X_val_stacked = np.hstack([
    X_val_proc,
    y_val_pred_lr,
    y_val_pred_lr_proba,
    explosive_val_preds,
    explosive_val_probas,
    y_val_pred_ridge
])

# ---------------------------------------------------
# Meta Model (Final)
# ---------------------------------------------------
# Define base model
base_xgb = XGBRegressor(
    objective='reg:tweedie',
    random_state=SEED,
    n_jobs=-1
)

# Wrap in MultiOutputRegressor
meta_model = MultiOutputRegressor(base_xgb, n_jobs=-1)

# Parameter grid (tight ranges around known good config)
param_dist = {
    "estimator__n_estimators": randint(100, 401),            # 100 to 400
    "estimator__learning_rate": uniform(0.01, 0.04),         # 0.01 to 0.05
    "estimator__max_depth": randint(3, 7),                   # 3 to 6
    "estimator__subsample": uniform(0.7, 0.2),               # 0.7 to 0.9
    "estimator__colsample_bytree": uniform(0.3, 0.6),        # 0.3 to 0.9
    "estimator__reg_alpha": uniform(0, 1.5),                 # 0 to 1.5
    "estimator__reg_lambda": uniform(0, 1.5),                # 0 to 1.5
    "estimator__gamma": uniform(0, 1)                      # 0 to 1
}

# Run RandomizedSearchCV
tuner = RandomizedSearchCV(
    estimator=meta_model,
    param_distributions=param_dist,
    n_iter=15,
    scoring='neg_mean_absolute_error',
    cv=3,
    random_state=SEED,
    n_jobs=-1
)

# Fit search
tuner.fit(X_train_stacked, y_train_regression)

# Get the tuned model
meta_model = tuner.best_estimator_

# Print best params
print("Best Parameters:\n", tuner.best_params_)

# Predict and evaluate
y_val_pred_meta = meta_model.predict(X_val_stacked)

# Collect metrics
metrics_list = []
for idx, target in enumerate(y_train_regression.columns):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = y_val_pred_meta[:, idx]
    metrics_list.append({
        'target': target,
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pinball_loss_under':  mean_pinball_loss(y_true, y_pred, alpha=0.1),
        'pinball_loss_over':  mean_pinball_loss(y_true, y_pred, alpha=0.9)
    })
    
meta_metrics_df = pd.DataFrame(metrics_list)
print("\nMeta-Model (XGBoost) Evaluation:")
print(meta_metrics_df)
meta_metrics_df.to_parquet("evaluation_metrics.parquet", index=False)

# Meta Model Feature Importance
base_feature_names = preprocessor.named_steps['transform'].get_feature_names_out(input_features=features).tolist()
base_feature_names = [name for i, name in enumerate(base_feature_names)
                      if preprocessor.named_steps['variance'].get_support()[i]]

# Stacked model output names
lr_bin_names      = [f"{t}_logreg_pred" for t in targets2]     # 12
lr_proba_names    = [f"{t}_logreg_proba" for t in targets2]     # 12
explosive_pred_names  = [f"{stat}_explosive_pred" for stat in explosive_thresholds.keys()]
explosive_proba_names = [f"{stat}_explosive_proba" for stat in explosive_thresholds.keys()]
lm_pred_names     = [f"{t}_linreg_pred" for t in targets2]      # 12

# Combine in correct stacking order
stacked_feature_names = (
    base_feature_names +
    lr_bin_names +
    lr_proba_names +
    explosive_pred_names +
    explosive_proba_names +
    lm_pred_names
)

# Final sanity check
assert len(stacked_feature_names) == X_train_stacked.shape[1], (
    f"Feature name misalignment: {len(stacked_feature_names)} names vs {X_train_stacked.shape[1]} columns"
)

# --- SHAP Summary Plots for Each Target ---
for i, target in enumerate(y_train_regression.columns):
    print(f"\nSHAP Summary Plot for: {target}")

    model = meta_model.estimators_[i]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_stacked)

    shap.summary_plot(
        shap_values,
        X_val_stacked,
        feature_names=stacked_feature_names,
        max_display=20,
        show=True
    )

for i, target in enumerate(y_val_regression.columns):
    residuals = y_val_regression[target].values - y_val_pred_meta[:, i]
    plt.figure(figsize=(6, 3))
    sns.histplot(residuals, kde=True, bins=30, color='orange')
    plt.axvline(0, linestyle='--', color='gray')
    plt.title(f"Residuals for {target}")
    plt.xlabel("y_true - y_pred")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# Save Models and Preprocessor
# ---------------------------------------------------
joblib.dump(kmeans_final, 'nba_player_clustering.joblib')
joblib.dump(pca_pipeline, 'pca_pipeline.joblib')
joblib.dump(preprocessor, "preprocessor_pipeline.joblib")
joblib.dump(ridge_mt, 'nba_secondary_model.joblib')
joblib.dump(multi_logreg, 'nba_clf_model.joblib')
joblib.dump(calibrated_estimators, 'calibrated_logreg_estimators.joblib')
joblib.dump(y_train_bins.columns.tolist(), 'calibrated_logreg_target_names.joblib')
joblib.dump(meta_model, 'nba_meta_model.joblib')

# Save calibrated explosive models and thresholds
for stat in explosive_thresholds.keys():
    model_key = f"{stat}_calibrated_model"
    if model_key in explosive_results:
        model = explosive_results[model_key]
        best_thresh = explosive_results[f"{stat}_best_thresh"]   
        # Save model
        joblib.dump(model, f"explosive_{stat}_calibrated_model.joblib")  
        # Save threshold
        joblib.dump(best_thresh, f"explosive_{stat}_threshold.joblib")
        print(f"Saved: explosive_{stat}_calibrated_model.joblib and explosive_{stat}_threshold.joblib")

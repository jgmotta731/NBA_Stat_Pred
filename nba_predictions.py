# ---------------------------------------------------
# NBA Predictions Pipeline for New Data
# Created on Apr 29, 2025
# Author: Jack Motta
# ---------------------------------------------------
import joblib, warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from nba_api.stats.static import players
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
from unidecode import unidecode
import re
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Load gamelogs, upcoming schedule, injury database, and betting odds
gamelogs = pd.read_parquet("nba_gamelogs.parquet")
schedule = pd.read_parquet("nba_schedule.parquet")
hoopr_odds = pd.read_parquet('hoopr_game_odds.parquet')

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

# Step 1: Filter only starter injuries
starter_injuries = injury_db[injury_db['norm'].isin(starters)].copy()

# Step 2: Merge with gamelogs on team and date
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

# Only keep 2022 season onwards
gamelogs = gamelogs[gamelogs["season"] >= 2021].copy().reset_index(drop=True)

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

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
# Define tqdm-parallel wrapper to track progress in joblib Parallel
class TqdmParallel(Parallel):
    def __init__(self, total=None, *args, **kwargs):
        self._total = total
        self._pbar = tqdm(total=total)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.update(self.n_completed_tasks - self._pbar.n)

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

# Lag features
def compute_lag_features(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    result[f'{col}_lag1'] = df.groupby(['athlete_display_name', 'season'])[col].shift(1)
    result[f'{col}_lag2'] = df.groupby(['athlete_display_name', 'season'])[col].shift(2)
    result[f'{col}_lag3'] = df.groupby(['athlete_display_name', 'season'])[col].shift(3)
    return result

lag_results = TqdmParallel(n_jobs=-1, total=len(rolling_cols))(
    delayed(compute_lag_features)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + lag_results, axis=1)

# Expanding means
def compute_expanding_mean(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    result[f'{col}_expanding_mean'] = (
        df.groupby(['athlete_display_name', 'season'])[col]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    return result

expanding_results = TqdmParallel(n_jobs=-1, total=len(rolling_cols))(
    delayed(compute_expanding_mean)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)

# Rolling EWM, momentum, z-score, rank
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

rolling_ewm_results = TqdmParallel(n_jobs=-1, total=len(rolling_cols))(
    delayed(compute_rolling_and_ewm_features)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + rolling_ewm_results, axis=1)

# EWM trend slopes
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
        results = TqdmParallel(n_jobs=-1, total=len(groups))(
            delayed(compute_ewm_trend_stats_for_player)(name, grp, col, span) for name, grp in groups
        )
        slope_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)
        for idxs, slopes in results:
            slope_arr[idxs] = slopes
        gamelogs[f'{col}_ewm_trend_slope_{span}'] = slope_arr

# Opponent EWM stats
opponent_stats = ['points', 'rebounds', 'assists', 'three_point_field_goals_made']
ewm_spans = [5, 9, 19]

def compute_opponent_ewm_allowed(stat, span):
    opponent_daily = (
        gamelogs
        .groupby(['opponent_team_abbreviation', 'season', 'game_date'])[stat]
        .sum()
        .reset_index()
        .sort_values(['opponent_team_abbreviation', 'season', 'game_date'])
    )
    ewm_col = f'opponent_ewm_{stat}_allowed_span{span}'
    opponent_daily[ewm_col] = (
        opponent_daily
        .groupby(['opponent_team_abbreviation', 'season'])[stat]
        .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    )
    return opponent_daily[['opponent_team_abbreviation', 'season', 'game_date', ewm_col]]

opponent_ewm_results = TqdmParallel(n_jobs=-1, total=len(opponent_stats)*len(ewm_spans))(
    delayed(compute_opponent_ewm_allowed)(stat, span)
    for stat in opponent_stats
    for span in ewm_spans
)

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
# Build Upcoming Gamelogs w/ Predictions
# ---------------------------------------------------
# compute player clusters
full = gamelogs.groupby('athlete_display_name')[numeric_features].mean().reset_index()
kmeans_final = joblib.load("nba_player_clustering.joblib")
pca_pipeline = joblib.load("pca_pipeline.joblib")
full['cluster_label'] = kmeans_final.predict(pca_pipeline.transform(full[numeric_features])) + 1
clusters = full[['athlete_display_name','cluster_label']].astype({'cluster_label':'float32'})
clusters.to_parquet('player_clusters.parquet',index=False)
gamelogs = gamelogs.merge(clusters,on='athlete_display_name',how='left')
features += ['cluster_label']
categorical_features += ['cluster_label']

# Add the upcoming games
latest = (
    gamelogs[gamelogs['season']==2025]
    .sort_values('game_date')
    .groupby('athlete_display_name')
    .tail(1)
)
sched = schedule.loc[
    (schedule.home_abbreviation!='TBD') &
    (schedule.away_abbreviation!='TBD')
].copy()
sched['home_away'] = 'home'
away = sched.copy()
away['home_away'] = 'away'
away[['home_abbreviation','away_abbreviation']] = away[['away_abbreviation','home_abbreviation']]
sched = pd.concat([sched,away],ignore_index=True).rename(columns={
    'home_abbreviation':'team_abbreviation',
    'away_abbreviation':'opponent_team_abbreviation'
})
sched = sched.merge(latest,on='team_abbreviation',how='left').dropna(subset=['athlete_display_name'])
sched = sched.drop(columns=[c for c in sched if c.endswith('_y')])
sched.columns = [c[:-2] if c.endswith('_x') else c for c in sched]
sched['game_date'] = pd.to_datetime(sched['game_date'],errors='coerce')

# ---------------- Load Models ----------------
# Core pipelines and models
pre = joblib.load("preprocessor_pipeline.joblib")
meta_model = joblib.load("nba_meta_model.joblib")
lm_mt = joblib.load("nba_secondary_model.joblib")
calibrated_estimators = joblib.load("calibrated_logreg_estimators.joblib")
target_names = joblib.load("calibrated_logreg_target_names.joblib")  # Optional if needed

# Explosive stat names (must match training phase)
explosive_stats = [
    "points", "rebounds", "assists", "three_point_field_goals_made", "steals", "blocks"
]

# Load calibrated explosive models and thresholds
calibrated_explosive_models = {}
explosive_thresholds = {}

for stat in explosive_stats:
    model_path = f"explosive_{stat}_calibrated_model.joblib"
    threshold_path = f"explosive_{stat}_threshold.joblib"

    calibrated_explosive_models[stat] = joblib.load(model_path)
    explosive_thresholds[stat] = joblib.load(threshold_path)

# ---------------- Process Input ----------------
ct = pre.named_steps['transform']
input_cols = list(ct.feature_names_in_)
Xp = sched[input_cols]
Xp_proc = pre.transform(Xp)

# ---------------- Stage 1: Base Predictions ----------------
# Classification Labels (bins)
c = np.column_stack([
    est.predict(Xp_proc) for est in calibrated_estimators
])

# Classification Probabilities
p = np.column_stack([
    est.predict_proba(Xp_proc)[:, 1] for est in calibrated_estimators
])

# Explosive Labels & Probabilities (with thresholds applied)
e_preds = []
e_probs = []

for stat in explosive_stats:
    model = calibrated_explosive_models[stat]
    threshold = explosive_thresholds[stat]

    proba = model.predict_proba(Xp_proc)[:, 1]
    pred = (proba > threshold).astype(int)

    e_probs.append(proba.reshape(-1, 1))
    e_preds.append(pred.reshape(-1, 1))

# Stack explosive outputs
e_pred_stack = np.hstack(e_preds)
e_proba_stack = np.hstack(e_probs)

# Regression outputs (secondary model)
s = lm_mt.predict(Xp_proc)

# ---------------- Stage 2: Meta Prediction ----------------
Xp_meta = np.hstack([
    Xp_proc,        # Preprocessed inputs
    c,              # Classification labels
    p,              # Classification probs
    e_pred_stack,   # Explosive binary predictions (thresholded)
    e_proba_stack,  # Explosive probabilities
    s               # Secondary regression output
])

# Predict with meta-model
raw_preds = meta_model.predict(Xp_meta)

# Rename to match expected output column names
pred_cols = [
    'predicted_three_point_field_goals_made',
    'predicted_rebounds',
    'predicted_assists',
    'predicted_steals',
    'predicted_blocks',
    'predicted_points'
]
# Convert to DataFrame and assign column names
raw_preds = pd.DataFrame(raw_preds, columns=pred_cols)

# Add predictions to sched
sched[pred_cols] = raw_preds

# Construct final output DataFrame
df = sched[[
    'athlete_display_name','athlete_position_abbreviation',
    'team_abbreviation','opponent_team_abbreviation',
    'game_date','home_away'
]].reset_index(drop=True)

df = pd.concat([df, raw_preds.reset_index(drop=True)], axis=1)
today = date.today()
start = pd.to_datetime(today - timedelta(days=today.weekday()))
end   = start + timedelta(days=6)
df = df[(df.game_date>=start)&(df.game_date<=end)]
df = df.sort_values('game_date').drop_duplicates('athlete_display_name',keep='first')
df['game_date'] = df['game_date'].dt.strftime('%Y-%m-%d')

# enrich with headshots
try:
    pdp = pd.DataFrame(players.get_active_players())
    pdp['headshot_url'] = pdp['id'].apply(lambda i:
        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{i}.png"
    )
    pdp['norm'] = pdp['full_name'].apply(lambda s: unidecode(s).title())
    df['norm'] = df['athlete_display_name'].apply(lambda s: unidecode(s).title())
    df = df.merge(pdp,on='norm',how='left').drop(columns=[
        'id','full_name','first_name','last_name','is_active','norm'
    ])
except Exception as e:
    print("Warning fetching headshots:",e)

# ---------------------------------------------------
# Scraping Odds
# ---------------------------------------------------
API_KEY = 'api_key'
SPORT = 'basketball_nba'
REGION = 'us'
BOOKMAKER = 'draftkings'
MARKETS = [
    'player_points', 'player_assists', 'player_rebounds',
    'player_steals', 'player_blocks', 'player_threes'
]

# Step 1: Get upcoming NBA events
events_url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'
params = {
    'apiKey': API_KEY,
    'regions': REGION,
    'markets': 'h2h',
    'oddsFormat': 'decimal'
}
response = requests.get(events_url, params=params)
response.raise_for_status()
events = response.json()

# Step 2: Fetch player props with player names and odds
all_props = []

for event in events:
    event_id = event['id']
    event_url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds'
    params = {
        'apiKey': API_KEY,
        'regions': REGION,
        'markets': ','.join(MARKETS),
        'oddsFormat': 'decimal',
        'bookmakers': BOOKMAKER
    }
    event_response = requests.get(event_url, params=params)
    if event_response.status_code != 200:
        print(f"Failed to fetch props for event {event_id}: {event_response.status_code}, {event_response.text}")
        continue
    event_data = event_response.json()

    for bookmaker in event_data.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            for outcome in market.get('outcomes', []):
                all_props.append({
                    'event': f"{event_data['home_team']} vs {event_data['away_team']}",
                    'market': market['key'],                  # e.g., player_points
                    'label': outcome['name'],                # Over / Under
                    'description': outcome.get('description'),  # Player name
                    'point': outcome.get('point'),
                    'price': outcome['price']
                })

# Step 3: Convert to DataFrame
betting_odds_df = pd.DataFrame(all_props)

# Step 4: Pivot and clean
od = betting_odds_df.copy()
od['market_label'] = od['label'] + " " + od['market'].str.replace('player_', '', regex=False).str.title()

pp = od.pivot_table(index='description', columns='market_label', values='price', aggfunc='first')
pt = od.pivot_table(index='description', columns='market_label', values='point', aggfunc='first')

pp.columns = [f"{c} - Price" for c in pp.columns]
pt.columns = [f"{c} - Point" for c in pt.columns]

pivot = pd.concat([pp, pt], axis=1).reset_index().rename(columns={'description': 'player_name'})

# Step 5: Select relevant columns only
stats = {c.split(" - ")[0].split(" ", 1)[1] for c in pivot.columns if " - " in c}
cols = ['player_name']
for stat in sorted(stats):
    for lbl in ['Over', 'Under']:
        for m in ['Price', 'Point']:
            cn = f"{lbl} {stat} - {m}"
            if cn in pivot.columns:
                cols.append(cn)
pivot = pivot[cols]

# Step 6: Normalize player names
pivot['player_name'] = (
    pivot['player_name']
         .str.replace(r'[^\w\s]', '', regex=True)
         .str.replace(r'\b(?:[IVX]+)$', '', regex=True)
         .str.strip()
)

# create normalized join‐key for df
df['norm'] = (
    df['athlete_display_name']
      .str.replace(r'[^\w\s]', '', regex=True)                  # remove punctuation
      .str.replace(r'\b(?:[IVX]+)$', '', regex=True)            # remove Roman numerals at end
      .str.strip()
)

# ---------------------------------------------------
# Final Output
# ---------------------------------------------------
# merge df with pivot
out = df.merge(
    pivot,
    left_on='norm',
    right_on='player_name',
    how='inner'
).drop(columns=['player_name','norm'])

# Define the list of columns to keep
columns_to_keep = [
    'athlete_display_name', 'athlete_position_abbreviation',
    'team_abbreviation', 'opponent_team_abbreviation', 'game_date',
    'home_away', 'predicted_three_point_field_goals_made',
    'predicted_rebounds', 'predicted_assists', 'predicted_steals',
    'predicted_blocks', 'predicted_points', 'headshot_url'
]

# Subset the DataFrame
out = out[columns_to_keep].copy()
out.to_parquet("nba_predictions.parquet", index=False)
print("Saved nba_predictions.parquet")

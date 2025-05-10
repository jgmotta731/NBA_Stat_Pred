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

warnings.filterwarnings("ignore")

# Load, instantiate, and generate predictions
gamelogs = pd.read_parquet("nba_gamelogs.parquet")
schedule = pd.read_parquet("nba_schedule.parquet")
betting_odds_df = pd.read_csv("NBA_Betting_Odds.csv")

# downcast numeric types
gamelogs[gamelogs.select_dtypes('float64').columns] = \
    gamelogs.select_dtypes('float64')\
            .apply(pd.to_numeric, downcast='float')
gamelogs[gamelogs.select_dtypes('int64').columns] = \
    gamelogs.select_dtypes('int64')\
            .apply(pd.to_numeric, downcast='integer')
        
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
gamelogs = gamelogs[gamelogs["minutes"] > 0].reset_index(drop=True)

# Ensure game_date is datetime
# Ensure game_date is datetime
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])

# Global temporal sort by player, date, season, and team
gamelogs = gamelogs.sort_values(['athlete_display_name', 'game_date', 'season', 'team_abbreviation']).reset_index(drop=True)

# Clean up plus_minus column
if 'plus_minus' in gamelogs.columns:
    gamelogs['plus_minus'] = (
        gamelogs['plus_minus']
        .astype(str)
        .str.replace(r'^\+', '', regex=True)
        .replace('None', np.nan)
        .astype(np.float32)
    )

# Convert booleans to numeric flags
bool_cols = ['starter', 'ejected', 'team_winner', 'is_playoff']
for col in bool_cols:
    if col in gamelogs.columns:
        gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)

# Create Playoff Indicator
gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)

# Compute days since last game per player
gamelogs['days_since_last_game'] = (
    gamelogs.groupby('athlete_display_name')['game_date']
    .diff()
    .dt.days
    .astype(np.float32)
)

# Safe division helper
def safe_divide(numerator, denominator, eps=1e-3):
    return numerator / (denominator + eps)

# Define rolling columns
rolling_cols = [
    'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
    'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
    'rebounds', 'assists', 'steals', 'blocks','points', 'minutes'
]

def compute_lag_features(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)

    # Lags
    result[f'{col}_lag1'] = df.groupby('athlete_display_name')[col].shift(1)
    result[f'{col}_lag2'] = df.groupby('athlete_display_name')[col].shift(2)

    return result
lag_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(delayed(compute_lag_features)(gamelogs, col) for col in rolling_cols)
gamelogs = pd.concat([gamelogs] + lag_results, axis=1)

# Rolling features per player
def compute_rolling_and_ewm_features(df, col):
    df = df.sort_values(['athlete_display_name', 'game_date'])
    result = pd.DataFrame(index=df.index)
    
    # EWM features
    ewm_spans = [4, 9]
    for span in ewm_spans:
        shifted = df.groupby('athlete_display_name')[col].shift(1)
        shift2 = df.groupby('athlete_display_name')[col].shift(2)

        ewm_mean = shifted.groupby(df['athlete_display_name']) \
                          .ewm(span=span, adjust=False).mean() \
                          .reset_index(level=0, drop=True)
        ewm_shift2_mean = shift2.groupby(df['athlete_display_name']) \
                                .ewm(span=span, adjust=False).mean() \
                                .reset_index(level=0, drop=True)

        result[f'{col}_ewm_mean_span{span}'] = ewm_mean
        result[f'{col}_ewm_momentum_span{span}'] = shifted - ewm_shift2_mean
        result[f'{col}_ewm_zscore_span{span}'] = (shifted - ewm_mean) / (ewm_mean.std() + 1e-6)

        # League-wide EWM rank
        temp_ewm = df[['game_date']].copy()
        temp_ewm[f'{col}_ewm_mean_span{span}'] = ewm_mean
        ewm_rank = (
            temp_ewm.groupby('game_date')[f'{col}_ewm_mean_span{span}']
                    .transform(lambda x: x.rank(pct=True))
                    .astype(np.float32)
        )
        result[f'{col}_ewm_mean_span{span}_rank'] = ewm_rank

    return result

# Compute features in parallel
rolling_ewm_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_rolling_and_ewm_features)(gamelogs, col) for col in rolling_cols
)

# Concatenate all results to original DataFrame
gamelogs = pd.concat([gamelogs] + rolling_ewm_results, axis=1)

# Trend slope per player
def compute_ewm_trend_stats_for_player(name, group, col, span):
    group = group.sort_values('game_date')
    values = group[col].shift(1).to_numpy()
    idx = group.index.to_numpy()
    slopes = np.full(len(values), np.nan, dtype=np.float32)
    
    # Manually compute exponentially weighted regression
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

for span in [4, 9]:
    for col in targets:
        groups = gamelogs.groupby('athlete_display_name')
        results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
            delayed(compute_ewm_trend_stats_for_player)(name, grp, col, span)
            for name, grp in groups
        )
        slope_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)

        for idxs, slopes in results:
            slope_arr[idxs] = slopes

        gamelogs[f'{col}_ewm_trend_slope_{span}'] = slope_arr

gamelogs['points_ewm_std_span9'] = (
    gamelogs
    .groupby('athlete_display_name')['points']
    .transform(lambda x: x.shift(1).ewm(span=9, adjust=False).std())
)

# Flag hot games: points >= 1 std dev above EWM mean
gamelogs['is_hot_game'] = (
    (gamelogs['points'] >= gamelogs['points_ewm_mean_span9'] + gamelogs['points_ewm_std_span9'])
    .fillna(False)
    .astype(int)
)

# Compute consecutive hot streaks
gamelogs['hot_streak'] = (
    gamelogs
    .sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['is_hot_game']
    .transform(lambda x:
        x.shift(1)
         .groupby((x.shift(1) != 1).cumsum())
         .cumcount()
    )
    .astype(np.int32)
)

# Flag cold games: points < 1 std dev below EWM mean
gamelogs['is_cold_game'] = (
    (gamelogs['points'] < gamelogs['points_ewm_mean_span9'] - gamelogs['points_ewm_std_span9'])
    .fillna(False)
    .astype(int)
)

# Compute consecutive cold streaks
gamelogs['cold_streak'] = (
    gamelogs
    .sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['is_cold_game']
    .transform(lambda x:
        x.shift(1)
         .groupby((x.shift(1) != 1).cumsum())
         .cumcount()
    )
    .astype(np.int32)
)

# Cleanup
gamelogs = gamelogs.drop(columns=['is_hot_game', 'is_cold_game'], axis=1)

# Filter Post-COVID
gamelogs = gamelogs[gamelogs["season"] >= 2022].copy().reset_index(drop=True)

# Drop Columns
gamelogs = gamelogs.drop(columns=[
    'plus_minus', 'ejected', 'did_not_play', 'team_winner', 'active',
    'season_type', 'starter'
    ])

lagged_rolling_features = [col for col in gamelogs.columns
                           if 'trend' in col or 'lag' in col 
                           or 'momentum' in col or 'streak' in col
                           or 'span' in col or 'ewm' in col]

numeric_features = lagged_rolling_features + ['days_since_last_game']
categorical_features = ["home_away", "athlete_position_abbreviation", "is_playoff"]
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
pre = joblib.load("preprocessor_pipeline.joblib")
meta_model = joblib.load("nba_meta_model.joblib")
lm_mt = joblib.load("nba_secondary_model.joblib")
calibrated_estimators = joblib.load("calibrated_logreg_estimators.joblib")
calibrated_explosive = joblib.load("calibrated_explosive_model.joblib")
correction_models = joblib.load("bias_correction_models.pkl")  # dict of LinearRegression

# ---------------- Process Input ----------------
ct = pre.named_steps['transform']
input_cols = list(ct.feature_names_in_)
Xp = sched[input_cols]
Xp_proc = pre.transform(Xp)

# ---------------- Stage 1: Predictions ----------------
# Classification Bins (calibrated class labels)
c = np.column_stack([est.predict(Xp_proc) for est in calibrated_estimators])

# Classification Probabilities
p = np.column_stack([est.predict_proba(Xp_proc)[:, 1] for est in calibrated_estimators])

# Explosive classification
e_pred = calibrated_explosive.predict(Xp_proc).reshape(-1, 1)
e_proba = calibrated_explosive.predict_proba(Xp_proc)[:, 1].reshape(-1, 1)

# Regression outputs (secondary model)
s = lm_mt.predict(Xp_proc)

# ---------------- Stage 2: Meta Prediction ----------------
# Align stacking
Xp_meta = np.hstack([Xp_proc, c, p, e_pred, e_proba, s])

# Predict with meta-model
raw_preds = meta_model.predict(Xp_meta)

# ---------------- Stage 3: Apply Bias Correction ----------------
# Correct each target using stored linear post-hoc correction
columns = list(correction_models.keys())
corrected_preds = pd.DataFrame(index=sched.index, columns=columns)

# Apply bias correction
corrected_preds = pd.DataFrame(index=sched.index, columns=columns)

for i, col in enumerate(columns):
    model = correction_models[col]
    corrected_preds[col] = model.predict(raw_preds[:, i].reshape(-1, 1))

# Rename to match expected output column names
pred_cols = [
    'predicted_three_point_field_goals_made',
    'predicted_rebounds',
    'predicted_assists',
    'predicted_steals',
    'predicted_blocks',
    'predicted_points'
]
corrected_preds.columns = pred_cols

# Add predictions to sched
sched[pred_cols] = corrected_preds

# Construct final output DataFrame
df = sched[[
    'athlete_display_name','athlete_position_abbreviation',
    'team_abbreviation','opponent_team_abbreviation',
    'game_date','home_away'
]].reset_index(drop=True)

df = pd.concat([df, corrected_preds.reset_index(drop=True)], axis=1)
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

# process betting odds
od = betting_odds_df.copy()
od['market_label'] = od['label'] + " " + od['market'].str.replace('player_','',regex=False).str.title()
pp = od.pivot_table(index='description',columns='market_label',values='price',aggfunc='first')
pt = od.pivot_table(index='description',columns='market_label',values='point',aggfunc='first')
pp.columns=[f"{c} - Price" for c in pp.columns]
pt.columns=[f"{c} - Point" for c in pt.columns]
pivot = pd.concat([pp,pt],axis=1).reset_index().rename(columns={'description':'player_name'})
stats = {c.split(" - ")[0].split(" ",1)[1] for c in pivot if " - " in c}
cols=['player_name']
for stat in sorted(stats):
    for lbl in ['Over','Under']:
        for m in ['Price','Point']:
            cn=f"{lbl} {stat} - {m}"
            if cn in pivot: cols.append(cn)
pivot = pivot[cols]

# create normalized join‐key for df
df['norm'] = (
    df['athlete_display_name']
      .str.replace(r'[^\w\s]', '', regex=True)                   # remove punctuation
      .str.replace(r'\b(?:[IVX]+)$', '', regex=True)            # remove Roman numerals at end
      .str.strip()
)

# normalize player_name in pivot
pivot['player_name'] = (
    pivot['player_name']
         .str.replace(r'[^\w\s]', '', regex=True)
         .str.replace(r'\b(?:[IVX]+)$', '', regex=True)
         .str.strip()
)

# merge df with pivot (using pivot, not pivot_df)
out = df.merge(
    pivot,
    left_on='norm',
    right_on='player_name',
    how='inner'
).drop(columns=['player_name','norm'])

# now out contains athlete_display_name plus all your pivot columns
out.to_parquet("nba_predictions.parquet", index=False)
print("Saved nba_predictions.parquet")
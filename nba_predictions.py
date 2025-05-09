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
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta
from unidecode import unidecode
import xgboost as xgb

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

    rolling_windows = [10, 20]
    ewm_spans = [3, 5, 7, 9]

    # Rolling features
    for window in rolling_windows:
        shifted = df.groupby('athlete_display_name')[col].shift(1)
        shift2 = df.groupby('athlete_display_name')[col].shift(2)

        roll_mean = shifted.groupby(df['athlete_display_name']) \
                           .rolling(window, min_periods=1).mean() \
                           .reset_index(level=0, drop=True)
        roll_std = shifted.groupby(df['athlete_display_name']) \
                          .rolling(window, min_periods=1).std() \
                          .reset_index(level=0, drop=True)
        roll_shift2_mean = shift2.groupby(df['athlete_display_name']) \
                                 .rolling(window, min_periods=1).mean() \
                                 .reset_index(level=0, drop=True)

        # Save rolling features
        result[f'{col}_rolling{window}'] = roll_mean
        result[f'{col}_rolling_std{window}'] = roll_std
        result[f'{col}_rolling_zscore{window}'] = (shifted - roll_mean) / (roll_std + 1e-6)
        result[f'{col}_rolling_momentum{window}'] = shifted - roll_shift2_mean

        # League-wide rolling rank
        temp_roll = df[['game_date']].copy()
        temp_roll[f'{col}_rolling_mean{window}'] = roll_mean
        roll_rank = (
            temp_roll.groupby('game_date')[f'{col}_rolling_mean{window}']
                     .transform(lambda x: x.rank(pct=True))
                     .astype(np.float32)
        )
        result[f'{col}_rolling_mean{window}_rank'] = roll_rank

    # EWM features
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

# Trend slope/R² per player
def compute_trend_stats_for_player(name, group, col, window):
    group = group.sort_values('game_date')
    values = group[col].shift(1).to_numpy()
    idx = group.index.to_numpy()
    slopes = np.full(len(values), np.nan, dtype=np.float32)
    r2_scores = np.full(len(values), np.nan, dtype=np.float32)

    for i in range(window - 1, len(values)):
        y = values[i - window + 1 : i + 1]
        x = np.arange(len(y)).reshape(-1, 1)
        mask = ~np.isnan(y)
        if np.count_nonzero(mask) >= 3 and not np.allclose(y[mask], y[mask][0]):
            try:
                model = LinearRegression().fit(x[mask], y[mask])
                y_pred = model.predict(x[mask])
                slopes[i] = model.coef_[0]
                r2_scores[i] = r2_score(y[mask], y_pred)
            except:
                pass

    return idx, slopes, r2_scores

for window in [5]:
    for col in targets:
        groups = gamelogs.groupby('athlete_display_name')
        results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
            delayed(compute_trend_stats_for_player)(name, grp, col, window)
            for name, grp in groups
        )
        slope_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)
        r2_arr = np.full(len(gamelogs), np.nan, dtype=np.float32)

        for idxs, slopes, r2s in results:
            slope_arr[idxs] = slopes
            r2_arr[idxs] = r2s

        gamelogs[f'{col}_trend_slope_{window}'] = slope_arr
        gamelogs[f'{col}_trend_r2_{window}'] = r2_arr

# Team/Opponent rolling stats
def compute_opponent_team_rolling(df, window):
    # make sure df is sorted by date for each team/opponent
    df = df.sort_values('game_date').reset_index(drop=True)
    result = pd.DataFrame(index=df.index)

    # Team/Opponent basic rolling stats
    for group_col, stat_col, label in [
        ("opponent_team_abbreviation","team_score","opponent_points_allowed"),
        ("opponent_team_abbreviation","three_point_field_goals_made","opponent_3pm_allowed"),
        ("opponent_team_abbreviation","field_goals_made","opponent_fgm_allowed"),
        ("opponent_team_abbreviation","rebounds","opponent_rebounds_allowed"),
        ("opponent_team_abbreviation","free_throws_attempted","opponent_free_throws_allowed"),
        ("team_abbreviation",         "team_score","team_score"),
        ("team_abbreviation",         "assists","team_assists"),
    ]:
        rolling_mean = (
            df.groupby(group_col)[stat_col]
              .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
              .astype(np.float32)
        )
        result[f"{label}_rolling{window}"] = rolling_mean

        # Add league-wide rank for that stat on each game_date
        temp_df = df[['game_date']].copy()
        temp_df[f"{label}_rolling{window}"] = rolling_mean

        rolling_rank = (
            temp_df.groupby('game_date')[f"{label}_rolling{window}"]
                   .transform(lambda x: x.rank(pct=True))
                   .astype(np.float32)
        )
        result[f"{label}_rolling{window}_rank"] = rolling_rank

    # Derived shooting percentages (no ranks here unless explicitly needed)
    result[f'fg_pct_rolling{window}'] = (
        safe_divide(df[f'field_goals_made_rolling{window}'],
                    df[f'field_goals_attempted_rolling{window}'])
        .clip(0, 1)
        .astype(np.float32)
    )
    result[f'ft_pct_rolling{window}'] = (
        safe_divide(df[f'free_throws_made_rolling{window}'],
                    df[f'free_throws_attempted_rolling{window}'])
        .clip(0, 1)
        .astype(np.float32)
    )
    result[f'three_pt_pct_rolling{window}'] = (
        safe_divide(df[f'three_point_field_goals_made_rolling{window}'],
                    df[f'three_point_field_goals_attempted_rolling{window}'])
        .clip(0, 1)
        .astype(np.float32)
    )
    result[f'efg_pct_rolling{window}'] = (
        safe_divide(
            df[f'field_goals_made_rolling{window}'] +
            0.5 * df[f'three_point_field_goals_made_rolling{window}'],
            df[f'field_goals_attempted_rolling{window}']
        )
        .clip(0, 1)
        .astype(np.float32)
    )

    return result


# then call exactly as before:
team_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_opponent_team_rolling)(gamelogs, window) for window in [10, 20]
)
gamelogs = pd.concat([gamelogs] + team_results, axis=1)

# Expanding features per season
def compute_expanding_features(df, col):
    # assumes df is already sorted by ['athlete_display_name','season','game_date']
    result = pd.DataFrame(index=df.index)

    # season‐level expanding mean & std (shift first to avoid leakage)
    exp_mean = (
        df.groupby(['athlete_display_name', 'season'])[col]
          .transform(lambda x: x.shift(1).expanding().mean())
          .astype(np.float32)
    )
    exp_std = (
        df.groupby(['athlete_display_name', 'season'])[col]
          .transform(lambda x: x.shift(1).expanding().std())
          .astype(np.float32)
    )
    zscore = ((df[col].shift(1) - exp_mean) / (exp_std + 1e-6)).astype(np.float32)

    # rolling vs. opponent (5‐game window)
    roll_vs_opp5 = (
        df.groupby(['athlete_display_name','opponent_team_abbreviation'])[col]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
          .astype(np.float32)
    )

    # expanding rank (percentile rank of a player's expanding mean vs others on that date)
    temp_df = df[['season', 'game_date']].copy()
    temp_df[f'{col}_exp_mean'] = exp_mean

    exp_rank = (
        temp_df.groupby(['season', 'game_date'])[f'{col}_exp_mean']
               .transform(lambda x: x.rank(pct=True))
               .astype(np.float32)
    )

    # Assign to result
    result[f'{col}_season_expanding_mean']   = exp_mean
    result[f'{col}_season_expanding_std']    = exp_std
    result[f'{col}_season_expanding_zscore'] = zscore
    result[f'{col}_rolling_vs_opp5']         = roll_vs_opp5
    result[f'{col}_season_expanding_rank']   = exp_rank

    return result

expanding_results = Parallel(n_jobs=-1, backend='loky', verbose=1)(
    delayed(compute_expanding_features)(gamelogs, col) for col in rolling_cols
)
gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)

# Flag hot games: points > season‐to‐date expanding mean
gamelogs['is_hot_game'] = (
    (gamelogs['points'] > gamelogs['points_season_expanding_mean'])
    .fillna(False)
    .astype(int)
)

# Compute the consecutive “hot” streak length
gamelogs['hot_streak'] = (
    gamelogs
    .sort_values(['athlete_display_name','game_date'])
    .groupby('athlete_display_name')['is_hot_game']
    .transform(lambda x: 
        x.shift(1)                        # only prior games
         .groupby((x.shift(1) != 1).cumsum())  # new run when prior wasn’t hot
         .cumcount()                      # count within each run
    )
    .astype(np.int32)
)

# Cleanup
gamelogs = gamelogs.drop('is_hot_game', axis=1)

# Filter Post-COVID
gamelogs = gamelogs[gamelogs["season"] >= 2022].copy().reset_index(drop=True)

# Drop Columns
gamelogs = gamelogs.drop(columns=[
    'plus_minus', 'ejected', 'did_not_play', 'team_winner', 'active',
    'season_type', 'starter'
    ])

lagged_rolling_features = [col for col in gamelogs.columns
                           if 'rolling' in col or 'trend' in col or 'lag' in col 
                           or 'expanding' in col or 'momentum' in col or 'hot_streak' in col
                           or 'span' in col]

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

# load pipeline & models
pre = joblib.load("preprocessor_pipeline.joblib")
m2, m3, meta = [
    joblib.load(fn) for fn in [
        "nba_secondary_model.joblib",
        "nba_clf_model.joblib",
        "nba_meta_model.joblib"
    ]
]

# align input columns to preprocessor
ct = pre.named_steps['transform']
input_cols = list(ct.feature_names_in_)
Xp = sched[input_cols]
Xp_proc = pre.transform(Xp)

# ---------------- FIXED stacking order ----------------
# generate multi-stage predictions in the SAME order used for training:
c = m3.predict(Xp_proc)    # 1) classification bins  :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
s = m2.predict(Xp_proc)    # 2) secondary regression
Xp_meta = np.hstack([Xp_proc, c, s])
# -------------------------------------------------------

# build DMatrix and get raw log‑means, then exponentiate
dm = xgb.DMatrix(Xp_meta)
logm_list = [est.get_booster().predict(dm, output_margin=True) for est in meta.estimators_]
logm_arr   = np.vstack(logm_list).T
final      = np.exp(logm_arr)

# assign multi-output preds
pred_cols = [
    'predicted_three_point_field_goals_made',
    'predicted_rebounds',
    'predicted_assists',
    'predicted_steals',
    'predicted_blocks',
    'predicted_points'
]
sched[pred_cols] = final

# build this-week dataframe
df = sched[[
    'athlete_display_name','athlete_position_abbreviation',
    'team_abbreviation','opponent_team_abbreviation',
    'game_date','home_away'
]].copy()
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(final,columns=pred_cols)],axis=1)
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
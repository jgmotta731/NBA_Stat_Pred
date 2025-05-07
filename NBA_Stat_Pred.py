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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import root_mean_squared_error, r2_score, precision_score, f1_score, recall_score, accuracy_score, silhouette_score, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
pd.set_option('display.max_columns', 10)

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
        
# Only keep 2022 season onwards
gamelogs = gamelogs[gamelogs["season"] >= 2022].copy()

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
gamelogs = gamelogs[gamelogs["minutes"] >= 15].reset_index(drop=True)

# Clean up
del valid_players, player_game_counts
gc.collect()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
gamelogs = gamelogs.sort_values(['game_date', 'athlete_display_name', 'team_abbreviation']).reset_index(drop=True)

# Clean up plus_minus column
if 'plus_minus' in gamelogs.columns:
    gamelogs['plus_minus'] = (
        gamelogs['plus_minus']
        .astype(str)                                 # ensure string for replace
        .str.replace(r'^\+', '', regex=True)         # remove leading '+'
        .replace('None', np.nan)                     # convert 'None' to np.nan
        .astype(np.float32)                          # final dtype conversion
        )
    
# Convert booleans to numeric
for col in ['starter', 'ejected', 'team_winner', 'is_playoff']:
    if col in gamelogs.columns:
        gamelogs[col] = gamelogs[col].fillna(False).astype(np.int32)

rolling_cols = [
    'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
    'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
    'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'steals', 'blocks',
    'turnovers', 'fouls', 'points', 'minutes', 'plus_minus'
]

gamelogs = gamelogs.sort_values(['game_date', 'season', 'athlete_display_name']).reset_index(drop=True)
gamelogs['is_playoff'] = gamelogs['season_type'].isin([3, 5]).astype(np.int32)
gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'])
gamelogs['days_since_last_game'] = gamelogs.groupby('athlete_display_name')['game_date'].diff().dt.days.astype(np.float32)

# Safe division to avoid errors
def safe_divide(numerator, denominator, eps=1e-3):
    return numerator / (denominator + eps)

# Rolling Features Function
def compute_rolling_features(col):
    result = pd.DataFrame(index=gamelogs.index)
    for window in [5, 20]:
        def apply_rolling(series):
            shifted = series.shift(1)
            roll_mean = shifted.rolling(window=window, min_periods=1).mean()
            roll_std = shifted.rolling(window=window, min_periods=1).std()
            shift2 = series.shift(2)
            roll_shift2_mean = shift2.rolling(window=window, min_periods=1).mean()
            ewm_mean = series.shift(1).ewm(span=window, adjust=False).mean()
            return pd.DataFrame({
                f'{col}_rolling{window}': roll_mean,
                f'{col}_rolling_std{window}': roll_std,
                f'{col}_rolling_zscore{window}': (shifted - roll_mean) / (roll_std + 1e-6),
                f'{col}_momentum{window}': shifted - roll_shift2_mean,
                f'{col}_rolling_cv{window}': roll_std / (roll_mean + 1e-6),
                f'{col}_consistency_index_rolling{window}': roll_mean / (roll_std + 1e-6),
                f'{col}_ewm_rolling{window}': ewm_mean,}, index=series.index)
        # Apply per player
        rolled = gamelogs[[col, 'athlete_display_name']].groupby(
            'athlete_display_name', group_keys=False)[col].apply(apply_rolling).reset_index(level=0, drop=True)
        # Add global season rank
        rolled[f'{col}_global_rolling_rank{window}'] = (
            rolled[f'{col}_rolling{window}']
            .groupby(gamelogs['season']).rank(method="average", pct=True).astype(np.float32))
        result = pd.concat([result, rolled], axis=1)
    return result
results = Parallel(n_jobs=-1, backend="loky", verbose=1)(delayed(compute_rolling_features)(col) for col in rolling_cols)
gamelogs = pd.concat([gamelogs] + results, axis=1)

# Trend Slope Calculation
def compute_trend_stats_for_player(name, group, col, window):
    group = group.sort_values("game_date")
    values = group[col].shift(1).to_numpy()
    indices = group.index.to_numpy()
    slopes = np.full(len(values), np.nan, dtype=np.float32)
    intercepts = np.full(len(values), np.nan, dtype=np.float32)
    r2_scores = np.full(len(values), np.nan, dtype=np.float32)
    for i in range(window - 1, len(values)):
        y = values[i - window + 1:i + 1]
        x = np.arange(len(y)).reshape(-1, 1)
        mask = ~np.isnan(y)
        if np.count_nonzero(mask) >= 3 and not np.allclose(y[mask], y[mask][0]):
            try:
                model = LinearRegression().fit(x[mask], y[mask])
                y_pred = model.predict(x[mask])
                slopes[i] = model.coef_[0]
                intercepts[i] = model.intercept_
                r2_scores[i] = r2_score(y[mask], y_pred)
            except Exception:
                continue
    return indices, slopes, intercepts, r2_scores

# Apply for each target column and window
for window in [5, 20]:
    for col in targets:
        results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(compute_trend_stats_for_player)(name, group, col, window)
            for name, group in gamelogs.groupby('athlete_display_name'))
        slope_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
        intercept_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
        r2_vals = np.full(len(gamelogs), np.nan, dtype=np.float32)
        for indices, slopes, intercepts, r2s in results:
            slope_vals[indices] = slopes
            intercept_vals[indices] = intercepts
            r2_vals[indices] = r2s
        gamelogs[f'{col}_trend_slope_{window}'] = slope_vals
        gamelogs[f'{col}_trend_intercept_{window}'] = intercept_vals
        gamelogs[f'{col}_trend_r2_{window}'] = r2_vals

def compute_opponent_team_rolling(window):
    df = pd.DataFrame(index=gamelogs.index)
    # Team/Opponent Stat Rolling Features
    for group_col, stat_col, label in [
        ("opponent_team_abbreviation", "team_score", "opponent_points_allowed"),
        ("opponent_team_abbreviation", "three_point_field_goals_made", "opponent_3pm_allowed"),
        ("opponent_team_abbreviation", "field_goals_made", "opponent_fgm_allowed"),
        ("opponent_team_abbreviation", "rebounds", "opponent_rebounds_allowed"),
        ("team_abbreviation", "team_score", "team_score"),
        ("team_abbreviation", "rebounds", "team_rebounds"),
        ("team_abbreviation", "turnovers", "team_turnovers"),
        ("team_abbreviation", "three_point_field_goals_made", "team_3pm"),
        ("team_abbreviation", "three_point_field_goals_attempted", "team_3pa"),
        ("team_abbreviation", "assists", "team_assists"),
        ]:
        df[f"{label}_rolling{window}"] = (
            gamelogs[[stat_col, group_col]]
            .groupby(group_col)[stat_col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            ).astype(np.float32)
    # Rankings
    df[f"opponent_defense_rank_rolling{window}"] = (
        df[f"opponent_points_allowed_rolling{window}"]
        .groupby(gamelogs["season"])
        .rank(method="average", pct=True)
        ).astype(np.float32)
    df[f"team_offense_rank_rolling{window}"] = (
        df[f"team_score_rolling{window}"]
        .groupby(gamelogs["season"])
        .rank(method="average", pct=True)
        ).astype(np.float32)
    # Derived stats (assist-to-turnover, efficiency, usage rate)
    df[f'assist_to_turnover_rolling{window}'] = safe_divide(
        gamelogs[f'assists_rolling{window}'],
        gamelogs[f'turnovers_rolling{window}']
        ).astype(np.float32)
    df[f'orb_pct_rolling{window}'] = safe_divide(
        gamelogs[f'offensive_rebounds_rolling{window}'],
        gamelogs[f'rebounds_rolling{window}']
        ).astype(np.float32)
    df[f'fg_pct_rolling{window}'] = safe_divide(
        gamelogs[f'field_goals_made_rolling{window}'],
        gamelogs[f'field_goals_attempted_rolling{window}']
        ).clip(0, 1).astype(np.float32)
    df[f'ft_pct_rolling{window}'] = safe_divide(
        gamelogs[f'free_throws_made_rolling{window}'],
        gamelogs[f'free_throws_attempted_rolling{window}']
        ).clip(0, 1).astype(np.float32)
    df[f'three_pt_pct_rolling{window}'] = safe_divide(
        gamelogs[f'three_point_field_goals_made_rolling{window}'],
        gamelogs[f'three_point_field_goals_attempted_rolling{window}']
        ).clip(0, 1).astype(np.float32)
    df[f'usage_proxy_rolling{window}'] = safe_divide(
        gamelogs[f'field_goals_attempted_rolling{window}'] +
        gamelogs[f'free_throws_attempted_rolling{window}'] +
        gamelogs[f'turnovers_rolling{window}'],
        gamelogs[f'minutes_rolling{window}']
        ).astype(np.float32)
    return df
team_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(delayed(compute_opponent_team_rolling)(window) for window in [5, 20])
gamelogs = pd.concat([gamelogs] + team_results, axis=1)

def compute_expanding_features(col):
    df = pd.DataFrame(index=gamelogs.index)
    sorted_gamelogs = gamelogs.sort_values('game_date')
    # Grouping
    group = sorted_gamelogs.groupby(['athlete_display_name', 'season'])[col]
    # Expanding mean and std (shifted to avoid leakage)
    expanding_mean = group.transform(lambda x: x.shift(1).expanding().mean()).astype(np.float32)
    expanding_std = group.transform(lambda x: x.shift(1).expanding().std()).astype(np.float32)
    df[f'{col}_season_expanding_mean'] = expanding_mean
    df[f'{col}_season_expanding_std'] = expanding_std
    # Z-score
    df[f'{col}_season_expanding_zscore'] = (
        (sorted_gamelogs[col].shift(1) - expanding_mean) / (expanding_std + 1e-6)
        ).astype(np.float32)
    # CV and Consistency
    df[f'{col}_season_expanding_cv'] = (expanding_std / (expanding_mean + 1e-6)).astype(np.float32)
    df[f'{col}_season_consistency_index'] = (expanding_mean / (expanding_std + 1e-6)).astype(np.float32)
    # Rank (global across dataset, not grouped — optional)
    df[f'{col}_season_expanding_rank'] = expanding_mean.rank(method='average', pct=True).astype(np.float32)
    # Rolling vs. Opponent
    df[f'{col}_rolling_vs_opp5'] = (
        sorted_gamelogs
        .groupby(['athlete_display_name', 'opponent_team_abbreviation'])[col]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        ).astype(np.float32)
    return df
expanding_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(delayed(compute_expanding_features)(col) for col in rolling_cols)
gamelogs = pd.concat([gamelogs] + expanding_results, axis=1)

gamelogs['rolling_win_streak'] = (
    gamelogs.sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['team_winner']
    .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
).astype(np.int32)

gamelogs['rolling_loss_streak'] = (
    gamelogs.sort_values(['athlete_display_name', 'game_date'])
    .groupby('athlete_display_name')['team_winner']
    .transform(lambda x: x.shift(1).groupby((x.shift(1) != 0).cumsum()).cumcount())
).astype(np.int32)

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------
gamelogs = gamelogs.drop(columns=[
    'plus_minus', 'ejected', 'did_not_play', 'team_winner', 'active',
    'season_type', 'starter'
    ])

lagged_rolling_features = [col for col in gamelogs.columns
                           if 'rolling' in col or 'trend' in col or 'lag' in col 
                           or 'expanding' in col or 'momentum' in col]

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

# ---------------------------------------------------
# PCA + Clustering (with Temporal Train/Test Split)
# ---------------------------------------------------
# Temporal Split
train_gamelogs = gamelogs[gamelogs['season'] < 2025].copy()

# Aggregate to Player Level Separately
train_player_summary = train_gamelogs.groupby('athlete_display_name')[numeric_features].mean().reset_index()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features)
])

# Fit PCA only on train
X_train_proc = preprocessor.fit_transform(train_player_summary[numeric_features])
pca = PCA(random_state=SEED)
X_train_pca = pca.fit_transform(X_train_proc)

# Scree plots
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components'); plt.grid(True); plt.show()

plt.figure(figsize=(10,6))
plt.bar(np.arange(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot'); plt.grid(True); plt.tight_layout(); plt.show()

# Fit PCA Pipeline
pca_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2, random_state=SEED))
])

X_cluster_train = pca_pipeline.fit_transform(train_player_summary[numeric_features])

# Step 6: Find k clusters on train
inertias = []
silhouette_scores = []
k_values = range(2, 13)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    kmeans.fit(X_cluster_train)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_train, kmeans.labels_))

# Plot
fig, ax1 = plt.subplots(figsize=(12,5))
color = 'tab:blue'
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia (Distortion)', color=color)
ax1.plot(k_values, inertias, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True)

ax2 = ax1.twinx(); color = 'tab:orange'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_values, silhouette_scores, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Elbow Method + Silhouette Score'); plt.tight_layout(); plt.show()

# Step 7: Final KMeans
kmeans_final = KMeans(n_clusters=3, random_state=SEED)
kmeans_final.fit(X_cluster_train)

# Step 8: Assign clusters
train_player_summary['cluster_label'] = kmeans_final.labels_ + 1

# Step 9: Visualize clusters
pca_2d_df = pd.DataFrame(X_cluster_train, columns=['PC1', 'PC2'])
pca_2d_df['cluster_label'] = train_player_summary['cluster_label'].values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_2d_df, x='PC1', y='PC2', hue='cluster_label',
    palette='tab10', s=70, edgecolor='k')
plt.title('PCA of Player Profiles Colored by Cluster', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(title='Cluster Label', fontsize=12, title_fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

# ---------------------------------------------------
# Add Cluster Labels to Gamelogs
# ---------------------------------------------------
# Keep only athlete name and cluster_label from train_player_summary (NOT from test!)
player_clusters = train_player_summary[['athlete_display_name', 'cluster_label']].copy()
player_clusters['cluster_label'] = player_clusters['cluster_label'].astype('float32')
player_clusters.to_parquet('player_clusters.parquet')

# Merge cluster labels into gamelogs
gamelogs = gamelogs.merge(player_clusters, on='athlete_display_name', how='left')

# Add cluster label to predictors
features += ['cluster_label', 'was_missing']
categorical_features += ['cluster_label', 'was_missing']

# Convert all int32 columns to float32 in gamelogs
gamelogs[gamelogs.select_dtypes('int32').columns] = \
    gamelogs.select_dtypes('int32').astype('float32')

gc.collect()

# ---------------------------------------------------
# Train/Validation Split
# ---------------------------------------------------
# Ensure uniqueness in lists
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))
features = list(dict.fromkeys(features))

train_df = gamelogs[gamelogs["season"] < 2025].copy()
val_df = gamelogs[gamelogs["season"] >= 2025].copy()

 
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
    ('variance', VarianceThreshold(threshold=0.01))
])

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

# ---------------------------------------------------
# Base Model
# ---------------------------------------------------
# Wrap LinearRegression in MultiOutputRegressor
lr_base = LinearRegression()
multi_lr = MultiOutputRegressor(lr_base)
multi_lr.fit(X_train_proc, y_train_regression)

# Predict on train and validation sets
y_train_pred_lm = multi_lr.predict(X_train_proc)
y_val_pred_lm = multi_lr.predict(X_val_proc)

# Evaluate on validation set
metrics_list = []
for idx, target in enumerate(y_train_regression.columns):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = y_val_pred_lm[:, idx]
    metrics_list.append({
        'target': target,
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)


# ---------------------------------------------------
# Secondary Targets Model
# ---------------------------------------------------
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made', 
            'free_throws_attempted', 'free_throws_made',
            'three_point_field_goals_attempted',
            'team_score']

y_train_regression2 = train_df[targets2]
y_val_regression2 = val_df[targets2]

# Define and fit Linear Regression
lr_mt = MultiOutputRegressor(LinearRegression())
lr_mt.fit(X_train_proc, y_train_regression2)

# Predict on train/val
y_train_pred_lm2 = lr_mt.predict(X_train_proc)
y_val_pred_lm2 = lr_mt.predict(X_val_proc)

# Evaluate
metrics_list = []

for idx, target in enumerate(targets2):
    y_true = y_val_regression2.iloc[:, idx]
    y_pred = y_val_pred_lm2[:, idx]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_list.append({
        'target': target,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

regression_metrics_df = pd.DataFrame(metrics_list)
print(regression_metrics_df)

# ---------------------------------------------------
# Bin Classification Model
# ---------------------------------------------------
bin_edges = {
    'points': [0, 13, np.inf],
    'assists': [0, 3, np.inf],
    'rebounds': [0, 5, np.inf],
    'steals': [0, 1, np.inf],
    'blocks': [0, 1, np.inf],
    'three_point_field_goals_made': [0, 2, np.inf]
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

y_train_bins = create_bins(train_df, targets, bin_edges)
y_val_bins = create_bins(val_df, targets, bin_edges)

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

# Define Logistic Regression classifier with regularization
logreg_clf = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',
    multi_class='multinomial',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

# Wrap in MultiOutputClassifier for multilabel support
multi_logreg = MultiOutputClassifier(logreg_clf, n_jobs=1)
multi_logreg.fit(X_train_proc, y_train_bins)

# Predict on validation set
y_val_pred_lr = multi_logreg.predict(X_val_proc)
y_train_pred_lr = multi_logreg.predict(X_train_proc)

# Evaluate
metrics_list = []
for idx, target in enumerate(y_val_bins.columns):
    y_true = y_val_bins.iloc[:, idx]
    y_pred = y_val_pred_lr[:, idx]
    metrics_list.append({
        'target': target,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# ---------------------------------------------------
# Naive Model (Only predicting the mean)
# ---------------------------------------------------
naive_preds = np.tile(y_train_regression.mean().values, (X_val_proc.shape[0], 1))

# Evaluate
naive_metrics_list = []
for idx, target in enumerate(y_train_regression.columns):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = naive_preds[:, idx]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    naive_metrics_list.append({
        'target': target,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })
naive_metrics_df = pd.DataFrame(naive_metrics_list)
print(naive_metrics_df)

# ---------------------------------------------------
# Stack Predictions
# ---------------------------------------------------
X_train_stacked = np.hstack([
    X_train_proc,
    y_train_pred_lr,   # Logistic Regression (classification bin predictions)
    y_train_pred_lm2,  # Linear Regression on targets2
    y_train_pred_lm    # Linear Regression on full y_train_regression
])

X_val_stacked = np.hstack([
    X_val_proc,
    y_val_pred_lr,
    y_val_pred_lm2,
    y_val_pred_lm
])

# ---------------------------------------------------
# Meta Model (Final)
# ---------------------------------------------------
meta_xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    gamma=0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Wrap in MultiOutputRegressor
meta_model = MultiOutputRegressor(meta_xgb, n_jobs=1)
meta_model.fit(X_train_stacked, y_train_regression)

# Predict and evaluate
y_val_pred_meta = meta_model.predict(X_val_stacked)

metrics_list = []
for idx, target in enumerate(y_train_regression.columns):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = y_val_pred_meta[:, idx]
    metrics_list.append({
        'target': target,
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    })

meta_metrics_df = pd.DataFrame(metrics_list)
print("\nMeta-Model (XGBoost) Evaluation:")
print(meta_metrics_df)
meta_metrics_df.to_parquet("evaluation_metrics.parquet", index=False)

# ---------------------------------------------------
# Save Models and Preprocessor
# ---------------------------------------------------
joblib.dump(kmeans_final, 'nba_player_clustering.joblib')
joblib.dump(preprocessor, "preprocessor_pipeline.joblib")
joblib.dump(multi_lr, 'nba_base_model.joblib')
joblib.dump(lr_mt, 'nba_secondary_model.joblib')
joblib.dump(multi_logreg, 'nba_clf_model.joblib')
joblib.dump(meta_model, 'nba_meta_model.joblib')
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
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
pd.set_option('display.max_columns', 20)

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
        
# Only keep 2021 season onwards
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

# Clean up
del valid_players, player_game_counts
gc.collect()

# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
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

# Elbow Plot
explained = pca.explained_variance_
components = np.arange(1, len(explained) + 1)
plt.figure(figsize=(10,6))
plt.bar(
    components[:100],            # only the first 100 bars
    explained[:100],
    width=0.8                    # widen the bars (default is ~0.8–1.0; increase if you like)
)
plt.xlim(-2, 100)                 # cap x-axis
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot (First 100 Components)')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

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
kmeans_final = KMeans(n_clusters=7, random_state=SEED)
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
train_df['explosive_game'] = (train_df['points'] >= 30).astype(int)
val_df = gamelogs[gamelogs["season"] >= 2025].copy()
val_df['explosive_game'] = (val_df['points'] >= 30).astype(int)

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
    ('variance', VarianceThreshold(threshold=0.001))
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

# Define and fit Linear Regression (no regularization)
lm_base = LinearRegression(n_jobs=-1)
lm_mt = MultiOutputRegressor(lm_base, n_jobs=-1)
lm_mt.fit(X_train_proc, y_train_regression2)

# Predict on train/val
y_train_pred_lm = lm_mt.predict(X_train_proc)
y_val_pred_lm = lm_mt.predict(X_val_proc)

# Evaluate
metrics_list = []

for idx, target in enumerate(targets2):
    y_true = y_val_regression2.iloc[:, idx]
    y_pred = y_val_pred_lm[:, idx]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_list.append({
        'target': target,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# ---------------------------------------------------
# Bin Classification Model
# ---------------------------------------------------
bin_edges = {
    'minutes': [0, 32, np.inf],
    'field_goals_attempted': [0, 12, np.inf],
    'field_goals_made': [0, 6, np.inf],
    'free_throws_attempted': [0, 3, np.inf],
    'free_throws_made': [0, 2, np.inf],
    'three_point_field_goals_attempted': [0, 5, np.inf],
    'points': [0, 15, np.inf],
    'assists': [0, 4, np.inf],
    'rebounds': [0, 6, np.inf],
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

# Define Logistic Regression classifier with regularization
logreg_clf = LogisticRegression(
    C=0.01,
    penalty='l2',
    solver='saga',
    max_iter=10000,
    random_state=42,
    n_jobs=-1
)

# Wrap in MultiOutputClassifier for multilabel support
multi_logreg = MultiOutputClassifier(logreg_clf, n_jobs=-1)
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

# Calibrate each base logistic regression model
calibrated_estimators = []

for i, (target, base_clf) in enumerate(zip(y_train_bins.columns, multi_logreg.estimators_)):
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method='isotonic',
        cv=3, n_jobs=-1
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
        'roc_auc': roc_auc_score(y_true, y_prob)
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# ---------------------------------------------------
# Explosive Game Classification
# ---------------------------------------------------
# Define base model
explosive_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Light tuning grid (around your current values)
param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 6),
    "min_samples_leaf": randint(5, 20),
    "max_features": ["sqrt", "log2", None]
}

# Run randomized search
explosive_search = RandomizedSearchCV(
    estimator=explosive_base,
    param_distributions=param_dist,
    n_iter=10,  # light search
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit
explosive_search.fit(X_train_proc, train_df["explosive_game"])
explosive_model = explosive_search.best_estimator_
print("Best params:", explosive_search.best_params_)

explosive_model.fit(X_train_proc, train_df['explosive_game'])

# Predict on validation set
y_val_explosive_true = val_df['explosive_game']
y_val_explosive_pred = explosive_model.predict(X_val_proc)
y_val_explosive_prob = explosive_model.predict_proba(X_val_proc)[:, 1]

# Evaluate
explosive_metrics = {
    'model': type(explosive_model).__name__,
    'accuracy': accuracy_score(y_val_explosive_true, y_val_explosive_pred),
    'precision': precision_score(y_val_explosive_true, y_val_explosive_pred, zero_division=0),
    'recall': recall_score(y_val_explosive_true, y_val_explosive_pred, zero_division=0),
    'f1': f1_score(y_val_explosive_true, y_val_explosive_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_val_explosive_true, y_val_explosive_prob),
    'confusion_matrix': confusion_matrix(y_val_explosive_true, y_val_explosive_pred).tolist()
}

# Display as DataFrame
explosive_metrics_df = pd.DataFrame([explosive_metrics])
print(explosive_metrics_df)

# Wrap the trained explosive_model (don't retrain from scratch)
calibrated_model = CalibratedClassifierCV(
    estimator=explosive_model,
    method='isotonic',  # or 'isotonic'
    cv=3               # 3-fold internal cross-validation
)

# Fit calibration on training data
calibrated_model.fit(X_train_proc, train_df['explosive_game'])
explosive_probs_calibrated = calibrated_model.predict_proba(X_val_proc)[:, 1]

# Evaluate calibrated predictions
precision, recall, thresholds = precision_recall_curve(y_val_explosive_true, explosive_probs_calibrated)

y_pred_binary = (explosive_probs_calibrated >= 0.05).astype(int)

# Compute F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

# Find the best threshold
best_idx = f1_scores.argmax()
best_thresh = thresholds[best_idx]

print(f"Best threshold: {best_thresh:.4f}")
print(f"Precision: {precision[best_idx]:.4f}")
print(f"Recall: {recall[best_idx]:.4f}")
print(f"F1 Score: {f1_scores[best_idx]:.4f}")

# Probabilities (preferred input for meta-model)
explosive_train_proba = calibrated_model.predict_proba(X_train_proc)[:, 1]
explosive_val_proba   = calibrated_model.predict_proba(X_val_proc)[:, 1]

# Predicted Labels
explosive_train_pred = (explosive_train_proba >= best_thresh).astype(int)
explosive_val_pred   = (explosive_val_proba >= best_thresh).astype(int)

# ---------------------------------------------------
# Stack Predictions
# ---------------------------------------------------
X_train_stacked = np.hstack([
    X_train_proc,
    y_train_pred_lr,
    y_train_pred_lr_proba,
    explosive_train_pred.reshape(-1, 1),
    explosive_train_proba.reshape(-1, 1),
    y_train_pred_lm
])

X_val_stacked = np.hstack([
    X_val_proc,
    y_val_pred_lr,
    y_val_pred_lr_proba,
    explosive_val_pred.reshape(-1, 1),
    explosive_val_proba.reshape(-1, 1),
    y_val_pred_lm
])

# ---------------------------------------------------
# Meta Model (Final)
# ---------------------------------------------------
# Define base model
base_xgb = XGBRegressor(
    objective='reg:tweedie',
    random_state=42,
    n_jobs=-1
)

# Wrap in MultiOutputRegressor
meta_model = MultiOutputRegressor(base_xgb, n_jobs=-1)

# Parameter grid (tight ranges around known good config)
param_dist = {
    "estimator__n_estimators": randint(250, 400),
    "estimator__learning_rate": uniform(0.03, 0.03),         # ~centered at 0.05
    "estimator__max_depth": randint(3, 5),                   # 3 to 4
    "estimator__subsample": uniform(0.7, 0.2),               # 0.7 to 0.9
    "estimator__colsample_bytree": uniform(0.3, 0.4),        # 0.3 to 0.7
    "estimator__reg_alpha": uniform(0.1, 1.0),               # 0.5 to 1.5
    "estimator__reg_lambda": uniform(0.1, 1.0),              # 0.5 to 1.5
    "estimator__gamma": uniform(0, 0.5)                      # 0 to 0.5
}

# Run RandomizedSearchCV
tuner = RandomizedSearchCV(
    estimator=meta_model,
    param_distributions=param_dist,
    n_iter=15,
    scoring='neg_mean_absolute_error',
    cv=3,
    random_state=42,
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

def quantile_loss(y_true, y_pred, tau=0.5):
    error = y_true - y_pred
    return 2 * np.mean(np.maximum(tau * error, (tau - 1) * error))

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
        'quantile_loss': quantile_loss(y_true, y_pred, tau=0.1)
    })
    
meta_metrics_df = pd.DataFrame(metrics_list)
print("\nMeta-Model (XGBoost) Evaluation:")
print(meta_metrics_df)
meta_metrics_df.to_parquet("evaluation_metrics.parquet", index=False)

# Store adjusted predictions and coefficients
y_val_adjusted = pd.DataFrame(index=y_val_regression.index)
correction_models = {}

for i, target in enumerate(y_val_regression.columns):
    y_true = y_val_regression[target].values
    y_pred = y_val_pred_meta[:, i]
    
    # Fit linear correction: y_true ~ y_pred
    bias_model = LinearRegression()
    bias_model.fit(y_pred.reshape(-1, 1), y_true)
    
    # Store adjusted predictions
    y_val_adjusted[target] = bias_model.predict(y_pred.reshape(-1, 1))
    correction_models[target] = bias_model

print("\n📊 Post-Hoc Bias Correction Metrics:\n")
metrics_list = []

for i, target in enumerate(y_val_regression.columns):
    y_true = y_val_regression[target].values
    y_pred_raw = y_val_pred_meta[:, i]
    y_pred_corrected = y_val_adjusted[target].values

    metrics_list.append({
        "target": target,
        "mae_raw": mean_absolute_error(y_true, y_pred_raw),
        "mae_corrected": mean_absolute_error(y_true, y_pred_corrected),
        "rmse_raw": root_mean_squared_error(y_true, y_pred_raw),
        "rmse_corrected": root_mean_squared_error(y_true, y_pred_corrected),
        "r2_raw": r2_score(y_true, y_pred_raw),
        "r2_corrected": r2_score(y_true, y_pred_corrected)
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.round(4))

df_preds_corrected = pd.DataFrame({
    "athlete_display_name": val_df["athlete_display_name"].values,
    "game_date": pd.to_datetime(val_df["game_date"]),
    "three_point_field_goals_made": val_df["three_point_field_goals_made"].values,
    "predicted_three_point_field_goals_made": y_val_adjusted["three_point_field_goals_made"].values,
    "rebounds": val_df["rebounds"].values,
    "predicted_rebounds": y_val_adjusted["rebounds"].values,
    "assists": val_df["assists"].values,
    "predicted_assists": y_val_adjusted["assists"].values,
    "steals": val_df["steals"].values,
    "predicted_steals": y_val_adjusted["steals"].values,
    "blocks": val_df["blocks"].values,
    "predicted_blocks": y_val_adjusted["blocks"].values,
    "points": val_df["points"].values,
    "predicted_points": y_val_adjusted["points"].values
})

# Base features
base_feature_names = preprocessor.named_steps['transform'].get_feature_names_out(input_features=features).tolist()
base_feature_names = [name for i, name in enumerate(base_feature_names)
                      if preprocessor.named_steps['variance'].get_support()[i]]
# Stacked model output names
lr_bin_names      = [f"{t}_logreg_pred" for t in targets2]     # 12
lr_proba_names    = [f"{t}_logreg_proba" for t in targets2]     # 12
explosive_names   = ["explosive_pred", "explosive_proba"]       # 2
lm_pred_names     = [f"{t}_linreg_pred" for t in targets2]      # 12

# Combine in correct stacking order
stacked_feature_names = (
    base_feature_names +
    lr_bin_names +
    lr_proba_names +
    explosive_names +
    lm_pred_names
)

# Final sanity check
assert len(stacked_feature_names) == X_train_stacked.shape[1], (
    f"Feature name misalignment: {len(stacked_feature_names)} names vs {X_train_stacked.shape[1]} columns"
)

# --- SHAP Summary Plots for Each Target ---
for i, target in enumerate(y_train_regression.columns):
    print(f"\n🔍 SHAP Summary Plot for: {target}")

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

# ---------------------------------------------------
# Save Models and Preprocessor
# ---------------------------------------------------
joblib.dump(kmeans_final, 'nba_player_clustering.joblib')
joblib.dump(pca_pipeline, 'pca_pipeline.joblib')
joblib.dump(preprocessor, "preprocessor_pipeline.joblib")
joblib.dump(lm_mt, 'nba_secondary_model.joblib')
joblib.dump(multi_logreg, 'nba_clf_model.joblib')
joblib.dump(calibrated_model, 'calibrated_explosive_model.joblib')
joblib.dump(calibrated_estimators, 'calibrated_logreg_estimators.joblib')
joblib.dump(y_train_bins.columns.tolist(), 'calibrated_logreg_target_names.joblib')
joblib.dump(meta_model, 'nba_meta_model.joblib')
joblib.dump(correction_models, 'bias_correction_models.pkl')
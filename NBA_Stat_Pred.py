"""
Created on Sun Apr 20 2025

@author: Jack Motta
"""

# ---------------------------------------------------
# Imports & Seed Setup
# ---------------------------------------------------

import os, random, joblib, gc, warnings
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score, precision_score, f1_score, recall_score, accuracy_score, silhouette_score, mean_absolute_error
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import MultiTaskLasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

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
valid_players = player_game_counts[player_game_counts >= 10].index
gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

# After filtering players ➔ filter low-minute games
gamelogs = gamelogs[gamelogs["minutes"] >= 20].reset_index(drop=True)

# Clean up
del valid_players, player_game_counts
gc.collect()

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

# ---------------------------------------------------
# EDA
# ---------------------------------------------------

# Aggregate stats by player (take the mean of numeric features)
player_summary = gamelogs.groupby('athlete_display_name')[numeric_features].mean().reset_index()

# Calculate per-player averages for targets
for target in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'three_point_field_goals_made']:
    player_summary[f'avg_{target}'] = gamelogs.groupby('athlete_display_name')[target].mean().values

# Define bin edges
bin_edges = {
    'points': [0, 10, 15, 20, 25, 30, np.inf],
    'assists': [0, 2, 4, 6, 8, np.inf],
    'rebounds': [0, 4, 6, 8, 10, 12, np.inf],
    'steals': [0, 1, 2, np.inf],
    'blocks': [0, 1, 2, np.inf],
    'three_point_field_goals_made': [0, 1, 2, 3, 4, np.inf]
}

# Bin them
for target, edges in bin_edges.items():
    player_summary[f'{target}_bin'] = pd.cut(
        player_summary[f'avg_{target}'],
        bins=edges,
        labels=[f'{edges[i]}-{edges[i+1]}' if edges[i+1] != np.inf else f'{edges[i]}+' for i in range(len(edges)-1)],
        right=False
    )

player_summary = player_summary.drop(columns=[f'avg_{target}' for target in 
                                              ['points', 'assists', 'rebounds', 
                                               'steals', 'blocks', 
                                               'three_point_field_goals_made']])

# Use PCA for visualizations
eda_preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ]), numeric_features)
    ])

eda_pipeline = Pipeline([
    ('pre', eda_preprocessor),
    ('pca', PCA(n_components=2, random_state=SEED))
])

X_eda = eda_pipeline.fit_transform(player_summary[numeric_features])

pca_df = pd.DataFrame(X_eda, columns=['PC1', 'PC2'])
for bin_col in ['points_bin', 'rebounds_bin', 'assists_bin', 'steals_bin', 
                'blocks_bin', 'three_point_field_goals_made_bin']:
    pca_df[bin_col] = player_summary[bin_col].values

# Plot each bin separately
bin_names = ['points_bin', 'rebounds_bin', 'assists_bin', 'steals_bin', 
             'blocks_bin', 'three_point_field_goals_made_bin']

# Plot
for bin_name in bin_names:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=bin_name, palette='tab10', 
                    s=50, edgecolor='k')
    plt.title(f'PCA of Player Summaries Colored by {bin_name.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14); plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(title=bin_name.replace("_", " ").title(), fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout(); plt.show()

# ---------------------------------------------------
# PCA + Clustering (with Temporal Train/Test Split)
# ---------------------------------------------------

# Step 1: Temporal Split
train_gamelogs = gamelogs[gamelogs['season'] < 2025].copy()

# Step 2: Aggregate to Player Level Separately
train_player_summary = train_gamelogs.groupby('athlete_display_name')[numeric_features].mean().reset_index()

# Step 3: Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features)
])

# Step 4: Fit PCA only on train
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

# Step 5: Fit PCA Pipeline
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
# Interactive Scatterplot of Players
# ---------------------------------------------------

# Create a DataFrame combining PCA + names + cluster
#pca_interactive_df = pca_2d_df.copy()
#pca_interactive_df['athlete_display_name'] = train_player_summary['athlete_display_name']

# Set Plotly to open in browser
#pio.renderers.default = 'browser'

# Force cluster_label to str, so Plotly treats it as categorical
#pca_interactive_df['cluster_label'] = pca_interactive_df['cluster_label'].astype(str)

# Scatter plot
#fig = px.scatter(pca_interactive_df, x='PC1', y='PC2', color='cluster_label',
#                 hover_data=['athlete_display_name'],
#                 title='Interactive PCA of Players Colored by Cluster',
#                 color_discrete_sequence=px.colors.qualitative.Set1, width=1000, 
#                 height=700)
#fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
#fig.update_layout(legend_title_text='Cluster Label', coloraxis_showscale=False)
#fig.show()

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

# ---------------------------------------------------
# Train/Validation Split for Feature Selection
# ---------------------------------------------------

# Ensure uniqueness in lists
numeric_features = list(dict.fromkeys(numeric_features))
categorical_features = list(dict.fromkeys(categorical_features))
features = list(dict.fromkeys(features))

train_df = gamelogs[gamelogs["season"] < 2025].copy()
val_df = gamelogs[gamelogs["season"] >= 2025].copy()

X_train = train_df[features]
X_val = val_df[features]

# Split targets
y_train_regression = train_df[targets]
y_val_regression = val_df[targets]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Impute numeric missing with median
        ('scaler', StandardScaler())                   # Scale numeric features after imputation
    ]), numeric_features),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute categoricals
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))  # One-hot encode
    ]), categorical_features)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)

selector = VarianceThreshold(threshold=0.0)
X_train_proc = selector.fit_transform(X_train_proc)
X_val_proc = selector.transform(X_val_proc)

# ---------------------------------------------------
# Feature Selection via Lasso
# ---------------------------------------------------

# Fit MultiTaskLasso
multi_task_lasso = MultiTaskLasso(
    alpha=0.05,  
    max_iter=5000,
    random_state=42
)

multi_task_lasso.fit(X_train_proc, y_train_regression)

# Predict on validation set
y_val_pred = multi_task_lasso.predict(X_val_proc)
metrics_list = []
for idx, target in enumerate(targets):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = y_val_pred[:, idx]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    metrics_list.append({
        'target': target,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse
    })
# Create a dataframe to display results
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# Get coefficients from MultiTaskLasso
coefs = multi_task_lasso.coef_  # shape: (n_targets, n_features)

# Feature names after preprocessing and variance threshold
feature_names_after_preprocessing = preprocessor.get_feature_names_out()
selected_feature_names = feature_names_after_preprocessing[selector.get_support()]

# Build feature_coef dataframe (just like you wanted)
feature_coef = pd.DataFrame({
    'Feature': selected_feature_names,
    'Coefficient': list(coefs.T)  # Each feature has a list of coefficients (one per target)
})

#print("Feature coefficients:")
#print(feature_coef)

# Build list of features where ANY target's coefficient ≠ 0
new_features = [
    selected_feature_names[i] 
    for i in range(len(selected_feature_names)) 
    if any(abs(coefs[:, i]) != 0)
]

#print("\nSelected features:")
#print(new_features)
print(f"\nTotal selected features: {len(new_features)}")

# Get the indices of the selected features
selected_indices = [
    i for i in range(len(selected_feature_names))
    if selected_feature_names[i] in new_features
]

joblib.dump(new_features, "selected_features.joblib")
joblib.dump(selected_indices, "selected_indices.joblib")


# Now subset X_train and X_val
X_train_selected = X_train_proc[:, selected_indices]
X_val_selected = X_val_proc[:, selected_indices]

y_train_pred = multi_task_lasso.predict(X_train_proc)
y_val_pred = multi_task_lasso.predict(X_val_proc)

# 2. Stack predictions onto features
X_train_stacked = np.hstack([X_train_selected, y_train_pred])
X_val_stacked = np.hstack([X_val_selected, y_val_pred])

print("New training data shape:", X_train_stacked.shape)
print("New test data shape:", X_val_stacked.shape)

# ---------------------------------------------------
# Neural Network Bin Classification
# ---------------------------------------------------

bin_edges = {
    'points': [0, 15, np.inf],
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

# Build the base Random Forest model
rf = RandomForestClassifier(
    n_estimators=500,          # Number of trees
    max_depth=6,               # Max tree depth
    class_weight='balanced',   # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1
)

multi_rf = MultiOutputClassifier(rf, n_jobs=-1)
multi_rf.fit(X_train_stacked, y_train_bins)

# Predict on validation set
y_val_pred = multi_rf.predict(X_val_stacked)

# Evaluation
metrics_list = []

for idx, target in enumerate(y_val_bins.columns):
    y_true = y_val_bins.iloc[:, idx]
    y_pred = y_val_pred[:, idx]
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_list.append({
        'target': target,
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1
    })

metrics_df = pd.DataFrame(metrics_list)
print("\nValidation Metrics per Target:")
print(metrics_df)
metrics_df.to_parquet("nba_clf_metrics.parquet", index=False)

# Stack predictions
y_train_pred = multi_rf.predict(X_train_stacked)
X_train_stacked = np.hstack([X_train_stacked, y_train_pred])
X_val_stacked = np.hstack([X_val_stacked, y_val_pred])

# Confirm shapes
print("New X_train shape:", X_train_stacked.shape)
print("New X_val shape:", X_val_stacked.shape)


# ---------------------------------------------------
# Naive Model (Only predicting the mean)
# ---------------------------------------------------

# For each target, predict the training mean (constant)
naive_preds = np.tile(y_train_regression.mean().values, (X_val_stacked.shape[0], 1))

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
# Neural Net Regression
# ---------------------------------------------------

# Set up RandomForestRegressor
rf_reg = RandomForestRegressor(
    n_estimators=500,          # Number of trees
    max_depth=6,              # You can tune this
    random_state=42,
    n_jobs=9,
    max_features=0.75,
    min_samples_split=2,
    min_samples_leaf=1
)

multi_rf_reg = MultiOutputRegressor(rf_reg, n_jobs=1)
multi_rf_reg.fit(X_train_stacked, y_train_regression)

# Predict on validation stacked features
y_val_pred_reg = multi_rf_reg.predict(X_val_stacked)

# Evaluate
metrics_list = []

for idx, target in enumerate(y_train_regression.columns):
    y_true = y_val_regression.iloc[:, idx]
    y_pred = y_val_pred_reg[:, idx]
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
regression_metrics_df.to_parquet("evaluation_metrics.parquet", index=False)

# ---------------------------------------------------
# Save Models and Preprocessor
# ---------------------------------------------------

joblib.dump(selector, 'variance_threshold_selector.joblib')
joblib.dump(kmeans_final, 'nba_player_clustering.joblib')
joblib.dump(preprocessor, "preprocessor_pipeline.joblib")
joblib.dump(multi_task_lasso, "multi_task_lasso_model.joblib")
joblib.dump(multi_rf, "multi_rf_classifier_model.joblib")
joblib.dump(multi_rf_reg, "multi_rf_regressor_model.joblib")
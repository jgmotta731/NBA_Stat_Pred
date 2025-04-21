"""
Created on Sun Apr 20 21:23:04 2025

@author: jgmot
"""

# ---------------------------------------------------
# Imports & Seed Setup
# ---------------------------------------------------
import os, random, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb
import warnings
import json
from nba_api.stats.static import players
from datetime import date, timedelta

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
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
gamelogs.to_parquet("nba_gamelogs_processed.parquet", index=False)

# ---------------------------------------------------
# Train/Validation Split + Recency Weights
# ---------------------------------------------------
train_df = gamelogs[gamelogs['season'] < 2025].copy()
val_df   = gamelogs[gamelogs['season'] >= 2025].copy()

train_df['recency_weight'] = (
    (train_df['game_date'] - train_df['game_date'].min()).dt.days + 1
)
train_df['recency_weight'] /= train_df['recency_weight'].max()

# ---------------------------------------------------
# Define Feature Groups
# ---------------------------------------------------
embedding_features   = ['player_id', 'team_id', 'opponent_id']
categorical_features = [
    'home_away', 'athlete_position_abbreviation', 'is_playoff',
    'ejected_lag1','starter_lag1','is_playoff_lag1',
    'ejected_lag2','starter_lag2','is_playoff_lag2',
    'ejected_lag3','starter_lag3','is_playoff_lag3',
    'team_winner_lag1','team_winner_lag2','team_winner_lag3',
    'team_winner_lag4','team_winner_lag5'
]
numeric_features = [
    f for f in features
    if f not in categorical_features + embedding_features
]
# ensure probability_features are included
for col in probability_features:
    if col not in numeric_features:
        numeric_features.append(col)

# ---------------------------------------------------
# Preprocessor (with imputation & ignore‐unknown)
# ---------------------------------------------------
# enforce training‐time dtypes
string_cats = ['home_away', 'athlete_position_abbreviation']
flag_cats   = [c for c in categorical_features if c not in string_cats]

# cast string cats
for col in string_cats:
    train_df[col] = train_df[col].astype(str)
# cast flag cats
for col in flag_cats:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(int)
# cast numeric
for col in numeric_features:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_features),
    ('cat', cat_pipe, categorical_features)
])

X_train = preprocessor.fit_transform(train_df[numeric_features + categorical_features])
y_train = train_df[targets].values.astype(np.float32)
X_val   = preprocessor.transform(val_df[numeric_features + categorical_features])
y_val   = val_df[targets].values.astype(np.float32)

player_ids_train = train_df['player_id'].values
team_ids_train   = train_df['team_id'].values
opp_ids_train    = train_df['opponent_id'].values
player_ids_val   = val_df['player_id'].values
team_ids_val     = val_df['team_id'].values
opp_ids_val      = val_df['opponent_id'].values
train_weights    = train_df['recency_weight'].values.astype(np.float32)

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
class NBAGameDataset(Dataset):
    def __init__(self, X, y, player_ids, team_ids, opponent_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.p = torch.tensor(player_ids, dtype=torch.long)
        self.t = torch.tensor(team_ids, dtype=torch.long)
        self.o = torch.tensor(opponent_ids, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.p[idx], self.t[idx], self.o[idx], self.y[idx]

train_loader = DataLoader(
    NBAGameDataset(X_train, y_train, player_ids_train, team_ids_train, opp_ids_train),
    batch_size=128, shuffle=True
)
val_loader = DataLoader(
    NBAGameDataset(X_val, y_val, player_ids_val, team_ids_val, opp_ids_val),
    batch_size=128, shuffle=False
)

# ---------------------------------------------------
# Model Definition
# ---------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class NBAStatPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_players, num_teams):
        super().__init__()
        self.player_embed = nn.Embedding(num_players, 16)
        self.team_embed   = nn.Embedding(num_teams, 8)
        self.opp_embed    = nn.Embedding(num_teams, 8)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + 16 + 8 + 8, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU()
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(64, 1) for name in targets
        })

    def forward(self, x, p, t, o):
        e_p = self.player_embed(p)
        e_t = self.team_embed( t)
        e_o = self.opp_embed(  o)
        x   = self.input_proj(x)
        x   = torch.cat([x, e_p, e_t, e_o], dim=1)
        x   = self.fc(x)
        return torch.cat([self.heads[name](x) for name in self.heads], dim=1)

model = NBAStatPredictor(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1],
    num_players=len(player_le.classes_),
    num_teams=len(team_le.classes_)
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------
def train(model, train_loader, val_loader, train_weights, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    weights_t = torch.tensor(train_weights, dtype=torch.float32)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        start = 0
        for xb, pb, tb, ob, yb in train_loader:
            yb_log = torch.log1p(yb)
            bs = xb.size(0)
            w  = weights_t[start:start+bs].to(xb.device)
            start += bs

            optimizer.zero_grad()
            preds = model(xb, pb, tb, ob)
            loss  = (criterion(preds, yb_log).mean(dim=1) * w).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bs

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, pb, tb, ob, yb in val_loader:
                yb_log = torch.log1p(yb)
                preds  = model(xb, pb, tb, ob)
                val_loss += criterion(preds, yb_log).mean().item() * xb.size(0)

        print(f"Epoch {epoch:02d} | Train: {total_loss/len(train_loader.dataset):.4f} | Val: {val_loss/len(val_loader.dataset):.4f}")

train(model, train_loader, val_loader, train_weights, epochs=20)

# ---------------------------------------------------
# Evaluation: RMSE + R2
# ---------------------------------------------------
def evaluate_to_parquet(model, val_loader, target_names, output_path="evaluation_metrics.parquet"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, pb, tb, ob, yb in val_loader:
            preds = model(xb, pb, tb, ob)
            y_true.append(yb.numpy())
            y_pred.append(torch.expm1(preds).numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # build metrics table
    records = []
    for i, name in enumerate(target_names):
        rmse = root_mean_squared_error(y_true[:, i], y_pred[:, i])
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        records.append({
            "target": name,
            "rmse":    round(rmse, 3),
            "r2":      round(r2,   3)
        })

    df = pd.DataFrame.from_records(records)
    df.to_parquet(output_path, index=False)
    print(f"Evaluation metrics saved to {output_path}")
    return df

metrics_df = evaluate_to_parquet(model, val_loader, targets)
print(metrics_df)

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

# ---------------------------------------------------
# 1. Save Preprocessor
# ---------------------------------------------------
joblib.dump(preprocessor, "preprocessor.pkl")

# ---------------------------------------------------
# 2. Save PyTorch Model
# ---------------------------------------------------
torch.save(model, "nba_model_full.pt")
torch.save(model.state_dict(), "nba_model_weights.pt")

# ---------------------------------------------------
# 3. Save XGBoost Bin Models
# ---------------------------------------------------
for target, clf in models.items():
    clf.save_model(f"bin_model_{target}.json")

# ---------------------------------------------------
# 4. Save LabelEncoders
# ---------------------------------------------------
joblib.dump(player_le, "player_encoder.pkl")
joblib.dump(team_le,   "team_encoder.pkl")
joblib.dump(opponent_le,"opponent_encoder.pkl")

# ---------------------------------------------------
# 5. Save Manual Bin Boundaries
# ---------------------------------------------------
with open("manual_bins.json", "w") as f:
    json.dump(manual_bins, f)

# 🏀 NBA Player-Stat Predictor

A full pipeline to predict NBA player stats for upcoming games using PyTorch and XGBoost.

---

## Project Highlights

- **Full pipeline**: Data ingestion → Feature engineering → Training → Prediction
- **Model architecture**: Neural network (PyTorch) + XGBoost classifiers (stat bin probabilities)
- **Rich features**: Rolling stats, opponent defense metrics, lag indicators, player/team embeddings
- **Automation ready**: Supports Windows Task Scheduler with daily 10:30 AM updates
- **Shiny dashboard**: Searchable predictions with player headshots and matchup info

---

## Directory Structure

```
nba-stat-predictor/
├── app.R
├── update_nba_data.R
├── NBA_Stat_Pred.py
├── weekly_predictions.py
├── README.md
```

---

## Feature Engineering

- Rolling averages: 3, 5, 10-game windows (points, rebounds, etc.)
- Lag features: e.g. `starter_lag1`, `ejected_lag2`, `team_winner_lag3`
- Opponent defense: rolling 3/5/10 game stats allowed
- Recency feature: `days_since_last_game`
- Label encodings: player, team, opponent → IDs
- XGBoost bin probabilities: added as features to PyTorch model

---

## Modeling Overview

- **XGBoost Classifiers**: Trained on binned stat targets
- **PyTorch Multi-Output Regressor**:
  - Inputs: numeric features + categorical embeddings
  - Targets: continuous stat predictions

---

## Prediction Pipeline

- Updated daily
- Latest available player stats + upcoming matchups
- Preprocess → Transform → Predict

---

## Shiny App

A companion **Shiny dashboard** built in R provides a user-friendly interface to view predictions:

- Upload `nba_predictions.parquet` and `evaluation_metrics.parquet`
  - Generated from the `weekly_predictions.py` and `NBA_Stat_Pred.py`
- Filter by player, team, or position
- View predictions by matchup and date
- Includes headshots and sortable stat columns
- View model performance metrics with brief explanation on how to interpret them.

Use this dashboard to gather insights with player prop bets and fantasy players.

---

## Automation

Use Windows Task Scheduler:

- Trigger: Daily at 10:30 AM
- Action:
  ```
  Program/script: C:\Users\<you>\anaconda3\envs\<env_name>\python.exe
  Add arguments: C:\Users\<you>\NBA\predict_next_games.py
  ```

---

## Author

Built by Jack Motta using Dall-E for AI generated logos, Python, PyTorch, XGBoost, and R's `hoopR` package for live game logs.

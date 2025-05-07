# 🏀 NBA Player Stat Prediction

Predict per-game NBA player stats to gain an edge in **sports betting**, **fantasy sports**, and **basketball analytics**.

This project integrates advanced machine learning, temporal validation, player clustering, and model stacking. A **Shiny app** provides interactive access to predictions and model insights.

---

## 📌 Project Goals

- Forecast individual player stats (points, rebounds, assists, etc.) for upcoming NBA games
- Generate predictive features using rolling averages, trends, and opponent metrics
- Create a full ML pipeline including preprocessing, PCA, clustering, regression, classification, and stacking
- Serve results via a user-friendly web app

---

## 🎯 Target Stats

- Points  
- Rebounds  
- Assists  
- Steals  
- Blocks  
- 3PT Field Goals Made

---

## 🧠 Methodology Overview

### 🔧 Data Processing
- NBA gamelogs from 2022 onward (`.parquet` format)
- Filter players with ≥20 games and ≥15 minutes per game
- Downcasting types for memory efficiency
- Remove DNP and missing targets

### 🏗️ Feature Engineering
- Rolling window stats (5 and 20 games): mean, std, z-score, momentum, etc.
- Trend slope, intercept, and R² via linear regression
- Opponent- and team-level rolling stats
- Expanding stats and win/loss streak indicators
- Categorical encoding and missing data flagging

### 📉 Dimensionality Reduction + Clustering
- Apply PCA on player-season summaries
- KMeans clustering (optimal K=3 via elbow + silhouette)
- Use cluster labels as model features

### 🔁 Model Stack
1. **Base Regression:** Linear models for continuous prediction  
2. **Secondary Regression:** Key contributing stats (e.g., FGA, minutes, team points)  
3. **Classification Layer:** Predict binned outcomes via logistic regression  
4. **Meta Model:** Final prediction via stacked features using `XGBoost`

### 🧪 Evaluation
- Metrics: RMSE, MAE, R² (regression), Accuracy, Precision, Recall, F1 (classification)
- Naive benchmark included (predict mean values)

---

## 💻 Shiny App: [NBA Prediction Tool](https://jmotta31.shinyapps.io/NBA_Prediction_Tool/)

### App Features:
- 📋 **Prediction Table**: View predicted player stats
- 🎯 **Betting Edge Tool**: Enter odds to compute implied probabilities and model-based edge
- 📊 **Evaluation Dashboard**: Displays key performance metrics

---

## 🛠 Tech Stack

**Python (ML Pipeline):**  
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`

**R (Web App):**  
- `shiny`, `reactable`, `tidyverse`

---

## 📁 Directory Structure

```bash
NBA_Stat_Pred/
├── NBA_Stat_Pred.py             # Main model training pipeline
├── nba_classes.py               # Class used in nba_predictions.py
├── nba_predictions.py           # Script for generating new game predictions
├── update_nba_data.R            # R script to update datasets
├── app.R                        # Shiny App frontend
├── nba_predictions.parquet      # Generated predictions for upcoming games, used for running the app locally
├── evaluation_metrics.parquet   # Model metrics, also used for running Shiny app locally
├── www/                         # Images used in the app
└── README.md                    # Project documentation

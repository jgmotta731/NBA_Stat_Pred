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
- Exponentially Weighted Moving Average stats (span of 4 and 9): mean, std, z-score, momentum, etc.
- Trend slope, and R² via linear regression
- Hot/Cold streak indicators
- Categorical encoding and missing data flagging

### 📉 Dimensionality Reduction + Clustering
- Apply PCA on player-season summaries
- KMeans clustering (optimal K=3 via elbow + silhouette)
- Use cluster labels as model features

### 🔁 Model Stack
1. **Regression Layer:** Linear models for continuous prediction on player box score statistics (including targets and beyond)
2. **Classification Layer:** Predict binned outcomes via logistic regression, then calibrate and add predicted bins and probabilities
3. **Explosive Game Classification Layer:** Predict whether a player will score more than 30 points or not, then calibrate and add predicted label and probabilities
4. **Meta Model:** Final prediction via stacked features using `XGBoost`
5. **Post Hoc Bias Correction:** Final predictions are passed through a set of bias-correction models to reduce systematic over/underestimation in key stats

### 🧪 Evaluation
- Final Metrics: RMSE, MAE, R², Quantile Loss (Tau=0.1)

---

## 💻 Shiny App: [NBA Prediction Tool](https://jmotta31.shinyapps.io/NBA_Prediction_Tool/)

### App Features:
- 📋 **Prediction Table**: View predicted player stats with a column selector
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
├── nba_predictions.py           # Script for generating new game predictions
├── update_nba_data.R            # R script to update datasets
├── app.R                        # Shiny App frontend
├── nba_predictions.parquet      # Generated predictions for upcoming games, used for running the app locally
├── evaluation_metrics.parquet   # Model metrics, also used for running Shiny app locally
├── www/                         # Images used in the app
└── README.md                    # Project documentation
```

## Future Additions

As of now, the model struggles with identifying when a role player or bench player will be given significantly more minutes due to the injury of a starter or star player. An flag indicator of whether a star player or starter is out or won't play will be feature engineered to the training data in the hopes that the model can recognize whether the role player/bench player will be given significantly more minutes and therefore have more points, rebounds, assists, etc.

## Disclaimer

This project provides predictive models for NBA player performance and is intended for **entertainment and informational purposes only**, including potential use in sports betting contexts.

However, **no prediction is guaranteed**, and betting always carries financial risk. The author makes **no representations or warranties** regarding the accuracy, reliability, or profitability of the models or predictions.

By using this project, you acknowledge that any betting decisions you make are **at your own risk**, and the author is **not liable** for any losses or damages arising from use of this code or its outputs.

Use responsibly and follow all local laws and regulations related to sports wagering.

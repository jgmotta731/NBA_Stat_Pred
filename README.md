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
- Filter players with ≥20 games and > 0 minutes per game
- Downcasting types for memory efficiency
- Remove DNP and missing targets

### 🏗️ Feature Engineering
- Exponentially Weighted Moving Average stats (span of 4 and 9): mean, std, z-score, momentum, etc.
- Whether a player's starter teammate is injured coming into the game via StatSurge NBA Injury Database
- Trend slope, and R² via linear regression
- Hot/Cold streak indicators
- Categorical encoding and missing data flagging

### 📉 Dimensionality Reduction + Clustering
- Apply PCA on player-season summaries
- K-Means clustering
- Use cluster labels as model features

### 🔁 Model Stack
1. **Regression Layer:** Linear models for continuous prediction on player box score statistics
2. **Classification Layer:** Predict binned outcomes, then calibrate and add predicted bins and probabilities
3. **Explosive Game Classification Layer:** Predict whether a player will score more than 30 points or not, then calibrate and add predicted label and probabilities
4. **Meta Model:** Final prediction via stacked features

### 🧪 Evaluation
- Final Metrics: RMSE, MAE, R², Pinball Loss (α=0.1 and α=0.9)

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
├── update_nba_data.R            # R script to update gamelogs and game schedule
├── app.R                        # Shiny App frontend
├── www/                         # Images used in the app
└── README.md                    # Project documentation
```

## Disclaimer

This project provides predictive models for NBA player performance and is intended for **entertainment and informational purposes only**, including potential use in sports betting contexts.

However, **no prediction is guaranteed**, and betting always carries financial risk. The author makes **no representations or warranties** regarding the accuracy, reliability, or profitability of the models or predictions.

By using this project, you acknowledge that any betting decisions you make are **at your own risk**, and the author is **not liable** for any losses or damages arising from use of this code or its outputs.

Use responsibly and follow all local laws and regulations related to sports wagering.

## License

This project is not licensed for public use or redistribution.  
All rights reserved © Jack Motta 2025.

The source code in this repository is proprietary.  
Unauthorized use, copying, or distribution is prohibited.

## Data Sources & Dependencies

This project uses external data and tools under their respective terms and licenses:

- [Odds API](https://the-odds-api.com/) — used under their Terms of Service, solely to filter out players with no available upcoming betting props. No odds data is displayed or redistributed.
- [StatSurge NBA Injury Database](https://statsurge.substack.com/) — data ownership and licensing remain with the original authors.
- [HoopR](https://github.com/rtelmore/hoopR) — used under the MIT License.

This project does not redistribute or resell any proprietary data. All rights to external datasets remain with their respective owners.

# ---------------------------------------------------
# NBA Predictions Pipeline for New Data
# Created on Apr 29, 2025
# Author: Jack Motta
# ---------------------------------------------------
import pandas as pd
from nba_classes import FullPredictionPipeline

# Load, instantiate, and generate predictions
gamelogs = pd.read_parquet("nba_gamelogs.parquet")
schedule = pd.read_parquet("nba_schedule.parquet")
betting_odds = pd.read_csv("NBA_Betting_Odds.csv")

pipeline = FullPredictionPipeline()
y_pred = pipeline.predict(gamelogs, schedule, betting_odds)
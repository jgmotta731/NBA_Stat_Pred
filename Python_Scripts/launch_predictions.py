# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 23:53:11 2025
@author: jgmot

This script is intended to launch the nba_predictions.py only when the NBA season
is ongoing, otherwise it won't run.
"""

from datetime import date
import subprocess
import sys

# ---- Config ----
PYTHON_PATH = r"C:\Users\jgmot\anaconda3\python.exe"
SCRIPT_PATH = r"C:\Users\jgmot\NBA_Prediction_Tool\run_predictions.py"
PROJECT_DIR = r"C:\Users\jgmot\NBA_Prediction_Tool"

# ---- Season guard (month/day only) ----
def in_season(today=None):
    if today is None:
        today = date.today()

    m, d = today.month, today.day

    # Season runs Oct 21 → Dec 31, then Jan 1 → Jun 22
    in_fall = (m == 10 and d >= 21) or (m in [11, 12])
    in_spring = (m in [1, 2, 3, 4, 5]) or (m == 6 and d <= 22)

    return in_fall or in_spring

def main():
    if not in_season():
        print("Out of NBA season — skipping run.")
        sys.exit(0)

    print("NBA season is active — running predictions…")
    subprocess.run([PYTHON_PATH, SCRIPT_PATH])

if __name__ == "__main__":
    main()

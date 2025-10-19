# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 23:53:11 2025
@author: jgmot

This script is intended to launch the nba_predictions.py only when the NBA season
is ongoing, otherwise it won't run.
"""

from datetime import date
import subprocess, sys, os

PYTHON_PATH = r"C:\Users\jgmot\anaconda3\python.exe"
PROJECT_DIR = r"C:\Users\jgmot\NBA_Prediction_Tool"
SCRIPT_PATH = rf"{PROJECT_DIR}\Python_Scripts\nba_predictions.py"

def in_season(today=None):
    today = today or date.today()
    m, d = today.month, today.day
    in_fall   = (m == 10 and d >= 21) or (m in [11, 12])
    in_spring = (m in [1, 2, 3, 4, 5]) or (m == 6 and d <= 22)
    return in_fall or in_spring

def main():
    if not in_season():
        print("Out of NBA season — skipping run.")
        sys.exit(0)

    print("NBA season is active — running predictions…")
    try:
        res = subprocess.run(
            [PYTHON_PATH, "-u", SCRIPT_PATH],
            cwd=PROJECT_DIR,              # <<< critical so relative paths land in project
            capture_output=True, text=True, check=True
        )
        if res.stdout: print(res.stdout, end="")
        if res.stderr: print(res.stderr, file=sys.stderr, end="")
    except subprocess.CalledProcessError as e:
        print(f"\n--- nba_predictions.py FAILED (exit {e.returncode}) ---", file=sys.stderr)
        if e.stdout: print("\n[child stdout]\n" + e.stdout, file=sys.stderr, end="")
        if e.stderr: print("\n[child stderr]\n" + e.stderr, file=sys.stderr, end="")
        sys.exit(e.returncode)

if __name__ == "__main__":
    if not os.path.exists(SCRIPT_PATH):
        print(f"Script not found: {SCRIPT_PATH}", file=sys.stderr); sys.exit(1)
    main()

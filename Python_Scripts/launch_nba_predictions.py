# -*- coding: utf-8 -*-
"""
Launches nba_predictions.py only during the active NBA season.
Streams output live when possible; works inside Spyder too.
"""

from datetime import date
import subprocess, sys, os

PYTHON_PATH  = r"C:\Users\jgmot\anaconda3\python.exe"
PROJECT_DIR  = r"C:\Users\jgmot\NBA_Prediction_Tool"
SCRIPT_PATH  = rf"{PROJECT_DIR}\Python_Scripts\nba_predictions.py"

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
        # Detect Spyder (no real stdout fileno)
        is_spyder = "SPYDER" in os.environ.get("TERM_PROGRAM", "").upper() or \
                    any("SPYDER" in a.upper() for a in sys.argv)

        if is_spyder:
            # Fallback: capture and print after completion (Spyder-safe)
            res = subprocess.run(
                [PYTHON_PATH, "-u", SCRIPT_PATH],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True
            )
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            if res.returncode != 0:
                print(f"\n--- nba_predictions.py FAILED (exit {res.returncode}) ---", file=sys.stderr)
                sys.exit(res.returncode)
        else:
            # Normal terminal: stream live output
            process = subprocess.Popen(
                [PYTHON_PATH, "-u", SCRIPT_PATH],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode != 0:
                print(f"\n--- nba_predictions.py FAILED (exit {process.returncode}) ---", file=sys.stderr)
                sys.exit(process.returncode)

        print("\n--- nba_predictions.py completed successfully ---")

    except Exception as e:
        print(f"\n--- Launcher error: {e} ---", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if not os.path.exists(SCRIPT_PATH):
        print(f"Script not found: {SCRIPT_PATH}", file=sys.stderr)
        sys.exit(1)
    main()
# NBA Player Predictions — End-to-End (R + Python)

Portfolio project that builds a daily NBA player-level prediction dataset and a Shiny app to explore it.  
Data is gathered and engineered in **R** (hoopR) and **Python** (nba_api + custom pipeline), modeled with a **Bayesian-style neural network**, and scheduled on Windows via **Task Scheduler**.

> **Status:** designed to run daily during the NBA season (Oct 21 – Jun 22) and skip the offseason.

---

## Highlights

- **Automated data refresh (R):** pulls player box scores, schedule, and builds lineup/on-off features; guarded by a month/day season window.
- **Feature & model pipeline (Python):** scraping/assembly → feature engineering → clustering → preprocessing → BNN training.
- **Daily predictions:** generates `predictions/nba_predictions_YYYY-MM-DD.parquet` for upcoming games; Shiny app hot-reloads.
- **Launchers for production:** small wrapper scripts handle season windows, logs, and lock files to avoid overlapping runs.
- **Shiny UI:** dark theme, sortable/filterable tables, and a lightweight implied-probability calculator.

---

## Repository layout

```
.
├─ R_Scripts/
│  ├─ update_nba_data.R              # Pulls hoopR data; builds lineup/on-off; writes parquet datasets/
│  └─ launch_update_nba_data.R       # Season guard + logs + lock; calls update_nba_data.R (Task Scheduler)
│
├─ Python_Scripts/                   # (or project root, depending on your layout)
│  ├─ scraping_loading.py            # Scrapes advanced data, merges with hoopR gamelogs + injuries, light FE
│  ├─ feature_engineering.py         # Main feature engineering block
│  ├─ clustering.py                  # KMeans clustering; adds cluster label as a feature
│  ├─ preprocessing.py               # Train/test split, imputers, encoding, pipelines, scalers
│  ├─ bnn.py                         # Bayesian-like NN (MC dropout for quantiles)
│  ├─ run_pipeline.py                # Orchestrates training end-to-end
│  ├─ nba_predictions.py             # Loads artifacts; produces daily predictions parquet (with Odds filter)
│  └─ launch_nba_predictions.py      # Season guard + logs + lock; calls nba_predictions.py (Task Scheduler)
│
├─ app.R                             # Shiny app; reactive polling for predictions + metrics (parquet)
├─ datasets/                         # Outputs from data refresh (e.g., gamelogs, schedule, on/off)
├─ predictions/                      # Daily predictions parquet (nba_predictions_YYYY-MM-DD.parquet)
├─ logs/                             # Timestamped logs + latest copy from launchers
└─ README.md
```

---

## How the pipeline runs (daily)

1) **Data refresh (R)**  
   - `R_Scripts/update_nba_data.R`  
     - Pulls player gamelogs via **hoopR**.  
     - Builds upcoming **schedule**.  
     - Derives **lineup stints** and **on/off** summaries (parallelized).  
     - Writes parquet files in `datasets/`.

   - `R_Scripts/launch_update_nba_data.R`  
     - Season window (month/day only), timestamped logs in `logs/`, and a lightweight lock file (prevents overlap).  
     - Called by **Windows Task Scheduler** every morning.

2) **Modeling & predictions (Python)**  
   - `scraping_loading.py` assembles a unified dataframe (nba_api + hoopR + injury flags).  
   - `feature_engineering.py` derives features; `clustering.py` adds a KMeans cluster label.  
   - `preprocessing.py` builds pipelines (imputers/encoders/scalers); `bnn.py` defines a Bayesian-style NN (MC dropout).  
   - `run_pipeline.py` trains the model; artifacts saved under `models/` and `pipelines/`.  
   - `nba_predictions.py` loads artifacts, scores upcoming games, filters by available props, and saves **`predictions/nba_predictions_YYYY-MM-DD.parquet`**.  
   - `launch_nba_predictions.py` wraps the run for scheduling (season window, logs, lock).

3) **Shiny app**  
   - `app.R` reads:
     - `predictions/nba_predictions_YYYY-MM-DD.parquet` via **reactivePoll** (detects new day/mtime/size).
     - `datasets/Evaluation_Metrics.parquet` via **reactiveFileReader**.  
   - No republish needed day-to-day; the app picks up the latest parquet.

---

## Quickstart

> This project uses both **R** and **Python**. Any recent R (≥4.3) and Python (≥3.9) is fine.

### R setup

```r
# From an R console:
install.packages(c("dplyr","arrow","hoopR","lubridate","tidyr","data.table","purrr","stringr","tibble","parallel"))
```

Run once at the project root to populate `datasets/`:

```bash
Rscript R_Scripts/update_nba_data.R
```

### Python setup

Create/activate an environment, then:

```bash
pip install numpy pandas pyarrow joblib scikit-learn torch unidecode nba_api
# (plus any others used in your scripts)
```

Train models:

```bash
python Python_Scripts/run_pipeline.py
```

Generate daily predictions (writes to `predictions/`):

```bash
python Python_Scripts/nba_predictions.py
```

> **Note:** any secrets (e.g., Odds API) should be managed outside the repo. Do not commit keys.

### Shiny app

Run locally:

```r
# from project root
shiny::runApp("app.R")
```

Environment overrides (optional):

- `PREDICTIONS_DIR` (default: `predictions`)
- `METRICS_PATH` (default: `datasets/Evaluation_Metrics.parquet`)

---

## Windows Task Scheduler (daily, in-season only)

Create **two** daily tasks (one for R data refresh, one for Python predictions). Each task:

- **Trigger:** Daily at a morning time you prefer.  
- **Action → Start a program:**
  - **Program/script:** `"<path to Rscript.exe>"` or `"<path to python.exe>"`
  - **Add arguments:** `"<full path to launcher>"`  
    - R: `"...\R_Scripts\launch_update_nba_data.R"`
    - Py: `"...\Python_Scripts\launch_nba_predictions.py"`
  - **Start in:** `"<project root>"` (e.g., `C:\Users\...\NBA_Prediction_Tool`)
- **Run whether user is logged on or not**
- **Run with highest privileges** (if needed for paths)

The launchers handle:
- **Season window:** Oct 21–Dec 31 and Jan 1–Jun 22 (month/day only).
- **Lock file:** avoids overlapping runs.
- **Logs:** timestamped files under `logs/` plus a `data_refresh_latest.log` or similar copy.

---

## Modeling notes (BNN)

- The network in `bnn.py` uses **MC dropout** at inference to approximate predictive uncertainty.  
- Quantile outputs (e.g., 10th/50th/90th) enable **prediction intervals** per target (Points, Assists, Rebounds, 3PTM, Steals, Blocks).  
- `clustering.py` provides a **player cluster label** (KMeans on PCA-compressed stats) as an additional feature so the model can learn archetypes.

---

## Shiny behavior

- **Predictions table** is driven by a `reactivePoll` keyed on **path + mtime + size** so the UI refreshes when a new-day parquet lands or the current file is rewritten.  
- **Metrics** use `reactiveFileReader` and refresh daily.  
- No redeploy needed; app reads the freshest files from disk.

---

## Disclaimers

- This project is for **portfolio and educational purposes**.  
- The predictions and any betting-related outputs are **not financial advice**.  
- Use responsibly and in accordance with local laws.

---

## License recommendation

Use **Apache License 2.0**. It:
- **Requires attribution** (prevents others from claiming the code as theirs),
- Includes a **broad “AS IS” disclaimer** (limits liability),
- Adds a **patent grant** (more protection).

Create a file named `LICENSE` with the standard Apache-2.0 text and a copyright line like:

```
Copyright (c) 2025 Jack Motta
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

(If you prefer something shorter, **MIT** is fine too and also includes a warranty disclaimer, but Apache-2.0 is generally stronger for attribution + patent coverage.)

---

## Repo hygiene

Add a `.gitignore` for R and Python artifacts (parquets/logs/models):

```
/datasets/
/predictions/
/logs/
/models/
/pipelines/
.Rhistory
.Rproj.user/
__pycache__/
*.pyc
.env
*.parquet
```

Do **not** commit API keys or private data.

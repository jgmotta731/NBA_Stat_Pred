# ---------------------------------------------------
# NBA Predictions Pipeline for New Data
# Created on Apr 29, 2025
# Author: Jack Motta
# ---------------------------------------------------
import os, warnings, joblib, requests, torch, random, sys, subprocess, datetime, time, re, json
import numpy as np
import pandas as pd
from datetime import date
from unidecode import unidecode
from nba_api.stats.static import players
from scipy.stats import norm
from Python_Scripts.scraping_loading import run_scrape_and_load
from Python_Scripts.feature_engineering import run_feature_engineering
from Python_Scripts.bnn import predict_mc, load_bnn_for_inference
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import commonteamroster
from pathlib import Path
from playwright.sync_api import sync_playwright

# ---------------------------------------------------
# Config
# ---------------------------------------------------
# Project root = one level above Python_Scripts
ROOT = Path.cwd()
DATASETS_DIR = str(ROOT / "datasets")
PREDICTIONS_DIR = str(ROOT / "predictions")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

SCHEDULE_PATH = str(Path(DATASETS_DIR) / "nba_schedule.parquet")
GAMELOGS_PATH = str(Path(DATASETS_DIR) / "nba_gamelogs.parquet")
warnings.filterwarnings("ignore")

# Reproducible MC inference (dropout + noise)
def set_inference_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # make GPU math deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def current_season_bounds(today=None):
    """
    Returns (start_year, end_year) for the current NBA season.
    Oct–Dec → season N–(N+1); Jan–Sep → (N-1)–N.
    """
    today = today or date.today()
    if today.month >= 10:   # Oct–Dec
        return today.year, today.year + 1
    else:                   # Jan–Sep
        return today.year - 1, today.year

def season_label(start_year, end_year):
    return f"{start_year}-{str(end_year)[-2:]}"

def build_seasons(first_start_year=2021):
    """
    Builds season labels from first_start_year up through the *current* season.
    When October arrives, it will automatically include the new season (e.g., 2025-26).
    """
    cur_start, cur_end = current_season_bounds()
    labels = []
    for y in range(first_start_year, cur_start + 1):
        labels.append(season_label(y, y + 1))
    return labels

SEASONS = build_seasons(first_start_year=2021)
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
PREPROCESSED_PATH = "datasets/gamelogs_ready_for_fe.parquet"
FEATURED_PATH     = "datasets/gamelogs_features.parquet"
SCHEDULE_PATH = f"{DATASETS_DIR}/nba_schedule.parquet"
GAMELOGS_PATH = f"{DATASETS_DIR}/nba_gamelogs.parquet"

set_inference_seed(42)
def main():
    # ---------------------------------------------------
    # Step 1: Scrape + load + preprocess (caches to PREPROCESSED_PATH)
    # ---------------------------------------------------
    gamelogs = run_scrape_and_load(
        seasons=SEASONS,
        preprocessed_path=PREPROCESSED_PATH
    )

    # ---------------------------------------------------
    # Step 2: Feature engineering (caches to FEATURED_PATH)
    # ---------------------------------------------------
    gamelogs, feature_groups = run_feature_engineering(
        gamelogs,
        save_path=FEATURED_PATH,
        cache_read=True,
        cache_write=True,
        id_col="GAME_ID"
    )

    # feature groups from training
    embedding_features   = feature_groups["embedding_features"]
    prior_features       = feature_groups["prior_features"]
    targets              = feature_groups["targets"]  # ['three_point_field_goals_made', ..., 'points']

    # ---------------------------------------------------
    # Step 3: upcoming schedule → build prediction rows
    # ---------------------------------------------------
    if not os.path.exists(SCHEDULE_PATH):
        raise FileNotFoundError(f"Schedule not found at {SCHEDULE_PATH}")
    schedule = pd.read_parquet(SCHEDULE_PATH)
    schedule["game_date"] = pd.to_datetime(schedule["game_date"], errors="coerce")
    
    _valid_sched = schedule.loc[
        (schedule["home_abbreviation"] != "TBD") &
        (schedule["away_abbreviation"] != "TBD")
    ].copy()
    
    # infer target season from earliest game on the slate
    first_gd = _valid_sched["game_date"].min()
    if pd.isna(first_gd):
        first_gd = pd.Timestamp.today()
    start_year = int(first_gd.year if first_gd.month >= 10 else first_gd.year - 1)
    season_str = f"{start_year}-{str(start_year + 1)[-2:]}"
    
    ROSTER_CACHE = Path(DATASETS_DIR) / f"roster_{season_str}.parquet"
    
    def _fetch_team_roster(tid: int, season: str, retries: int = 4, timeout: int = 30, pause: float = 1.2):
        """Robust fetch for one team with retries + backoff. Returns DataFrame or None."""
        wait = pause
        for attempt in range(1, retries + 1):
            try:
                df = commonteamroster.CommonTeamRoster(team_id=tid, season=season, timeout=timeout).get_data_frames()[0]
                if not df.empty:
                    return df
                # if empty, try again
            except Exception as e:
                if attempt == retries:
                    print(f"[ROSTER] failed tid={tid} after {retries} tries: {e}")
                    return None
            time.sleep(wait)
            wait *= 1.8  # exponential backoff
    
    # 1) Try to load cached roster
    if ROSTER_CACHE.exists():
        roster = pd.read_parquet(ROSTER_CACHE)
    else:
        teams_meta = pd.DataFrame(static_teams.get_teams())[['id', 'abbreviation']].rename(
            columns={'id': 'team_id', 'abbreviation': 'team_abbreviation'}
        )
        roster_parts = []
        failed_tids = []
    
        # polite rate limit + retries
        for _, row in teams_meta.iterrows():
            tid = int(row['team_id'])
            df = _fetch_team_roster(tid, season_str, retries=5, timeout=45, pause=1.0)
            if df is None:
                failed_tids.append(tid)
                continue
            df = df[['PLAYER_ID', 'PLAYER', 'POSITION']].copy()
            df['team_id'] = tid
            roster_parts.append(df)
            time.sleep(0.6)  # small delay between calls to avoid throttling
    
        if roster_parts:
            roster = pd.concat(roster_parts, ignore_index=True)
            roster = (
                roster.merge(teams_meta, on='team_id', how='left')
                      .rename(columns={
                          'PLAYER': 'athlete_display_name',
                          'PLAYER_ID': 'player_id',
                          'POSITION': 'athlete_position_abbreviation'
                      })
                      [['team_abbreviation','athlete_display_name','athlete_position_abbreviation','player_id']]
            )
        else:
            roster = pd.DataFrame(columns=['team_abbreviation','athlete_display_name',
                                           'athlete_position_abbreviation','player_id'])
    
        # Fallback fill for teams that failed: use last-known players from your gamelogs
        if failed_tids:
            # map team_id -> abbreviation
            tid2abbr = teams_meta.set_index('team_id')['team_abbreviation'].to_dict()
            latest_any = (
                gamelogs.sort_values('game_date')
                        .groupby(['athlete_display_name'], as_index=False)
                        .tail(1)
            )
            # Bring in last-known team_abbreviation to filter by team for fallback
            last_by_team = latest_any[['athlete_display_name','athlete_position_abbreviation','team_abbreviation']].copy()
    
            fallback_rows = []
            for tid in failed_tids:
                abbr = tid2abbr.get(tid)
                if not abbr:
                    continue
                # take up to 18 most recent players for that team (won't be perfect, but prevents empty teams)
                temp = last_by_team[last_by_team['team_abbreviation'] == abbr].head(18).copy()
                if temp.empty:
                    continue
                temp['player_id'] = np.nan
                fallback_rows.append(temp[['team_abbreviation','athlete_display_name','athlete_position_abbreviation','player_id']])
    
            if fallback_rows:
                fallback_block = pd.concat(fallback_rows, ignore_index=True)
                roster = pd.concat([roster, fallback_block], ignore_index=True).drop_duplicates(
                    ['team_abbreviation','athlete_display_name'], keep='first'
                )
    
        # cache whatever we assembled so we don’t refetch next run
        roster.to_parquet(ROSTER_CACHE, index=False)
    
    # 2) Build the team slate (home/away doubled)
    sched = _valid_sched.copy()
    sched["home_away"] = "home"
    _away = sched.copy()
    _away["home_away"] = "away"
    _away[["home_abbreviation","away_abbreviation"]] = _away[["away_abbreviation","home_abbreviation"]]
    sched = (
        pd.concat([sched, _away], ignore_index=True)
          .rename(columns={"home_abbreviation": "team_abbreviation",
                           "away_abbreviation": "opponent_team_abbreviation"})
    )
    
    # 3) Expand to one row per rostered player (now we include traded/rookie players)
    sched = sched.merge(roster, on="team_abbreviation", how="left")
    
    # 4) Attach priors by player
    latest_by_player = (
        gamelogs.sort_values("game_date")
                .groupby("athlete_display_name", as_index=False)
                .tail(1)
    )
    
    # Normalize player names for merging
    def _norm_name(s: str) -> str:
        if pd.isna(s):
            return ""
        s = str(s)
        s = re.sub(r"\(.*?\)", " ", s)                # drop parentheticals like "(DEN)"
        s = unidecode(s)                               # remove accents: Dončić -> Doncic
        s = s.replace("’", "'")                        # unify apostrophes
        s = s.lower()
        s = re.sub(r"\b(jr|sr|jr\.|sr\.|ii|iii|iv|v)\b", " ", s)  # strip suffixes
        s = re.sub(r"[^\w\s]", " ", s)                 # remove punctuation (.,-,')
        s = re.sub(r"\s+", " ", s).strip()             # collapse whitespace
        return s
    
    sched = sched.assign(norm_name=lambda d: d["athlete_display_name"].apply(_norm_name))
    latest_by_player = latest_by_player.assign(norm_name=lambda d: d["athlete_display_name"].apply(_norm_name))
    
    sched = sched.merge(
        latest_by_player.drop(columns=["team_abbreviation","athlete_display_name"], errors="ignore"),
        on="norm_name", 
        how="left", 
        suffixes=("", "_latest")
    ).drop(columns=["norm_name"], errors="ignore")
    
    # 5) Clean and finalize
    sched = sched.drop(columns=[c for c in sched.columns if c.endswith("_y")], errors="ignore")
    sched.columns = [c[:-2] if c.endswith("_x") else c for c in sched.columns]
    sched = sched.dropna(subset=["athlete_display_name"]).reset_index(drop=True)

    # ---------------------------------------------------
    # Step 4: load preprocessing artifacts & build tensors
    # ---------------------------------------------------
    preprocessor    = joblib.load("pipelines/preprocessor_pipeline.joblib")
    prior_pipeline  = joblib.load("pipelines/prior_pipeline.joblib")
    embed_encoder   = joblib.load("pipelines/embed_encoder.joblib")
    embedding_sizes = joblib.load("pipelines/embedding_sizes.joblib")
    
    # use GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # embeddings (IDs were +1/clamped in training)
    for col in embedding_features:
        sched[col] = sched[col].astype(str)
    Xp_embed = embed_encoder.transform(sched[embedding_features]).astype(np.int64) + 1
    for j, (num_embeddings, _) in enumerate(embedding_sizes):
        Xp_embed[:, j] = np.clip(Xp_embed[:, j], 0, num_embeddings - 1)
    Xp_embed_tensor = torch.tensor(Xp_embed, dtype=torch.long, device=device)
    
    # numeric block via training preprocessor
    Xp = sched[preprocessor.named_steps["transform"].feature_names_in_]
    Xp_proc_base = preprocessor.transform(Xp)
    Xp_proc = pd.DataFrame(Xp_proc_base, columns=preprocessor.get_feature_names_out())
    
    # priors (expanding_mean/std for the 6 targets, already in prior_features order)
    Xp_prior = prior_pipeline.transform(sched[prior_features]).astype(np.float32)
    Xp_prior_tensor = torch.tensor(Xp_prior, dtype=torch.float32, device=device)
    
    # final numeric tensor (drop embed cols; there is no 'target' column at inference)
    Xp_num_tensor = torch.tensor(
        Xp_proc.drop(columns=[c for c in embedding_features if c in Xp_proc.columns], errors="ignore").values,
        dtype=torch.float32,
        device=device
    )

    # ---------------------------------------------------
    # Step 5: BNN model (same architecture as training) + weights
    # ---------------------------------------------------
    model = load_bnn_for_inference(
        weights_path="models/bnn/nba_bnn_weights_only.pt",
        input_dim=Xp_num_tensor.shape[1],
        embedding_sizes=embedding_sizes,
        prior_dim=Xp_prior_tensor.shape[1],
        output_dim=len(targets),
        aux_dim=len(targets),
        device=device,
    )
    model.train()  # enable dropout for MC sampling
    
    # ---- load calibration scales (STD + Quantile), aligned to `targets` order
    calib_df = pd.read_parquet("datasets/Calibration_Scales.parquet")
    calib_vec = (
        calib_df.set_index("Target")
                .reindex(targets)["Aleatoric_Scale"]
                .astype("float32")
                .to_numpy()
    )
    calib_t = torch.tensor(calib_vec, dtype=torch.float32, device=device).view(1, -1)
    
    # quantile scales (optional; fall back to 1.0 if missing)
    try:
        qcal_df = pd.read_parquet("datasets/Calibration_Scales.parquet")
        q_vec = (
            qcal_df.set_index("Target")
                   .reindex(targets)["Quantile_Scale"]
                   .astype("float32")
                   .to_numpy()
        )
    except Exception:
        print("Note: Calibration scales not found; using 1.0 (no adjustment).")
        q_vec = np.ones(len(targets), dtype=np.float32)
    q_vec_r = q_vec.reshape(1, -1)
    
    # ---------------------------------------------------
    # Step 6: MC-dropout predictions (STD-calibrated), then quantile-calibrate, then invert scaling
    # ---------------------------------------------------
    # ---- MC predictions for future games (STD-calibrated via alea_scale) ----
    mean_pred, std_epi, std_ale, std_pred, q10, q50, q90 = predict_mc(
        model=model,
        X_num=Xp_num_tensor,
        X_embed=Xp_embed_tensor,
        prior_tensor=Xp_prior_tensor,
        T=50,
        alea_scale=calib_t,  # STD calibration applied here
    )
    
    # ---- quantile calibration (in standardized space) ----
    q10 = mean_pred + (q10 - mean_pred) * q_vec_r
    q90 = mean_pred + (q90 - mean_pred) * q_vec_r
    # median left as-is (symmetry)
    
    # ---- inverse transform to original units ----
    y_scaler = joblib.load("pipelines/y_scaler.joblib")
    mean_unscaled   = y_scaler.inverse_transform(mean_pred)
    median_unscaled = y_scaler.inverse_transform(q50)
    lower_unscaled  = y_scaler.inverse_transform(q10)
    upper_unscaled  = y_scaler.inverse_transform(q90)
    
    # stds to original units (already STD-calibrated by alea_scale in MC)
    std_epi_unscaled  = std_epi  * y_scaler.scale_
    std_ale_unscaled  = std_ale  * y_scaler.scale_
    std_pred_unscaled = std_pred * y_scaler.scale_
    
    # ---- std-based 80% bounds (two-sided) ----
    critval_80 = norm.ppf(0.9)
    std80_lower_unscaled = mean_unscaled - critval_80 * std_pred_unscaled
    std80_upper_unscaled = mean_unscaled + critval_80 * std_pred_unscaled
    
    # ---- assemble prediction frame ----
    pred_df = pd.DataFrame(mean_unscaled, columns=[f"{t}_mean" for t in targets])
    pred_df[[f"{t}_median"          for t in targets]] = median_unscaled
    pred_df[[f"{t}_lower"           for t in targets]] = lower_unscaled
    pred_df[[f"{t}_upper"           for t in targets]] = upper_unscaled
    pred_df[[f"{t}_std_pred"        for t in targets]] = std_pred_unscaled
    pred_df[[f"{t}_std_epistemic"   for t in targets]] = std_epi_unscaled
    pred_df[[f"{t}_std_aleatoric"   for t in targets]] = std_ale_unscaled
    pred_df[[f"{t}_std80_lower"     for t in targets]] = std80_lower_unscaled
    pred_df[[f"{t}_std80_upper"     for t in targets]] = std80_upper_unscaled
    pred_df[[f"{t}_pi80_width"      for t in targets]] = (upper_unscaled - lower_unscaled)
    
    pred_df["athlete_display_name"] = sched["athlete_display_name"].values
    pred_df["game_id"] = sched.get("game_id", pd.Series(index=sched.index, dtype="object"))
    
    # ---------------------------------------------------
    # Step 7: assemble output rows
    # ---------------------------------------------------
    # keep sched clean; DO NOT concat pred_df here (avoids duplicate column names)
    sched = sched.reset_index(drop=True)
    
    df = sched[[
        "athlete_display_name", "athlete_position_abbreviation",
        "team_abbreviation", "opponent_team_abbreviation",
        "game_date", "home_away"
    ]].reset_index(drop=True)
    
    # add the prediction columns (without duplicate key columns)
    df = pd.concat(
        [df, pred_df.drop(columns=["athlete_display_name", "game_id"], errors="ignore").reset_index(drop=True)],
        axis=1
    )
    
    # if any duplicate column names slipped in, drop the duplicates
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # choose the earliest upcoming game_date >= today (or fallback to last available)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    today_ts = pd.Timestamp.today().normalize()
    future_mask = df["game_date"] >= today_ts
    
    if future_mask.any():
        next_date = df.loc[future_mask, "game_date"].min()
        df = df[df["game_date"] == next_date].copy()
        print(f"Using next slate: {next_date.date()} (rows={len(df)})")
    else:
        # no future games in schedule — keep the most recent available date instead
        if df["game_date"].notna().any():
            last_date = df["game_date"].max()
            df = df[df["game_date"] == last_date].copy()
            print(f"No future games; using last available slate: {last_date.date()} (rows={len(df)})")
        else:
            print("No valid game_date values; keeping df as-is.")
    
    df = df.sort_values("game_date").drop_duplicates("athlete_display_name", keep="first")
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")

    # headshots
    try:
        pdp = pd.DataFrame(players.get_active_players())
        pdp["headshot_url"] = pdp["id"].apply(lambda i: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{i}.png")
        pdp["norm"] = pdp["full_name"].apply(lambda s: unidecode(s).title())
        df["norm"] = df["athlete_display_name"].apply(lambda s: unidecode(s).title())
        df = df.merge(pdp, on="norm", how="left").drop(columns=["id","full_name","first_name","last_name","is_active","norm"])
    except Exception as e:
        print("Warning fetching headshots:", e)

    # ---------------------------------------------------
    # Step 8: odds scrape for filtering
    # ---------------------------------------------------
    # ------------------ Parsing ------------------
    def _extract_names_from_offer_categories(offer_categories) -> set:
        names = set()
        for cat in (offer_categories or []):
            if (cat.get("name") or "").lower() != "player props":
                continue
            for desc in cat.get("offerSubcategoryDescriptors", []) or []:
                sub = desc.get("offerSubcategory") or {}
                for offer_list in sub.get("offers", []) or []:
                    for offer in offer_list:
                        for oc in offer.get("outcomes", []) or []:
                            nm = oc.get("participant") or oc.get("description")
                            nn = _norm_name(nm)
                            if nn:
                                names.add(nn)
        return names
    
    # ------------------ Source A: DraftKings via Playwright (robust) ------------------
    def _get_dk_names_playwright(timeout_s=30, headless=True) -> set:
        """
        Open DK NBA 'Player Props' page in a real Chromium context, capture ANY
        'eventgroups?...format=json' network response, parse, and return names.
        """
        league_urls = [
            "https://sportsbook.draftkings.com/leagues/basketball/nba?category=player-props",
            "https://sportsbook.draftkings.com/leagues/basketball/nba",  # backup
        ]
        names = set()
    
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            ctx = browser.new_context(
                locale="en-US",
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/128.0.0.0 Safari/537.36"),
            )
    
            # collect ANY matching responses
            captured = []
    
            def on_response(resp):
                url = resp.url or ""
                if "eventgroups" in url and "format=json" in url:
                    try:
                        data = resp.json()
                        if data:
                            captured.append(data)
                    except Exception:
                        pass
    
            ctx.on("response", on_response)
            page = ctx.new_page()
    
            # try a couple of URLs; wait a bit on each for JS to hydrate & requests to fire
            for u in league_urls:
                try:
                    page.goto(u, wait_until="domcontentloaded", timeout=timeout_s*1000)
                    # let the SPA hydrate and fire network calls
                    page.wait_for_timeout(3500)
                    # sometimes props need scroll to trigger lazy loads
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1500)
                    if captured:
                        break
                except Exception:
                    continue
    
            # If still nothing captured, try to fetch from inside the page (with cookies)
            if not captured:
                try:
                    api_url = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/84240?format=json"
                    data = page.evaluate("""async (url) => {
                        try {
                            const r = await fetch(url, {credentials:'include'});
                            if (!r.ok) return null;
                            return await r.json();
                        } catch(e){ return null; }
                    }""", api_url)
                    if data:
                        captured.append(data)
                except Exception:
                    pass
    
            browser.close()
    
        # parse all captured blobs (if multiple, union names)
        for data in captured:
            eg = (data.get("eventGroup") or {})
            cats = eg.get("offerCategories", [])
            names |= _extract_names_from_offer_categories(cats)
    
        return names
    
    # ------------------ Source B: PrizePicks (public JSON) ------------------
    def _get_prizepicks_names(timeout_s=20) -> set:
        """
        Public PrizePicks projections. We only need names present today.
        """
        url = "https://api.prizepicks.com/projections?league_id=7"  # NBA historically ID=7
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        incl = {str(i["id"]): i for i in data.get("included", [])}
        names = set()
        for proj in data.get("data", []):
            rel = proj.get("relationships", {}) or {}
            pl = rel.get("new_player", {}).get("data") or {}
            pid = str(pl.get("id", ""))
            nm = incl.get(pid, {}).get("attributes", {}).get("name", "")
            nn = _norm_name(nm)
            if nn:
                names.add(nn)
        return names
    
    # ------------------ Unified, cached, strict fetch ------------------
    def get_bettable_names_or_die(cache_dir="datasets", use_cache=True, headless=True) -> set:
        """
        Returns the union of DK (browser) + PrizePicks names.
        Caches daily. Raises if both sources fail (filtering is mandatory).
        """
        cp = Path(cache_dir) / f"prop_names_{date.today().isoformat()}.json"
        if use_cache and cp.exists():
            try:
                return set(json.load(open(cp, "r")))
            except Exception:
                pass
    
        names = set()
        errors = []
    
        # Source A: DK via Playwright (most aligned with sportsbook props)
        try:
            dk = _get_dk_names_playwright(headless=headless)
            names |= dk
        except Exception as e:
            errors.append(f"DK:{e}")
    
        # Source B: PrizePicks fallback
        try:
            pp = _get_prizepicks_names()
            names |= pp
        except Exception as e:
            errors.append(f"PP:{e}")
    
        if names:
            try:
                cp.parent.mkdir(parents=True, exist_ok=True)
                json.dump(sorted(list(names)), open(cp, "w"))
            except Exception:
                pass
            return names
    
        # nothing worked → fail clearly
        raise RuntimeError("Prop name fetch failed ({}). Filtering required but no source succeeded."
                           .format("; ".join(errors) or "no details"))
    
    bettable_names = get_bettable_names_or_die(cache_dir="datasets", use_cache=True, headless=True)
    df = df.copy()
    df["norm_name"] = df["athlete_display_name"].apply(_norm_name)
    out = df[df["norm_name"].isin(bettable_names)].drop(columns=["norm_name"])
    if out.empty:
        raise RuntimeError("Fetched prop names but none matched your players. Check normalization/schedule date.")

    # ---------------------------------------------------
    # Step 9: save parquet
    # ---------------------------------------------------
    cols_to_keep = [
        "athlete_display_name", "athlete_position_abbreviation",
        "team_abbreviation", "opponent_team_abbreviation",
        "game_date", "home_away", "headshot_url",
    
        "three_point_field_goals_made_mean", "three_point_field_goals_made_lower",
        "three_point_field_goals_made_upper", "three_point_field_goals_made_median",
        "three_point_field_goals_made_std_pred", "three_point_field_goals_made_std_epistemic",
        "three_point_field_goals_made_std_aleatoric", "three_point_field_goals_made_std80_lower",
        "three_point_field_goals_made_std80_upper", "three_point_field_goals_made_pi80_width",
    
        "rebounds_mean", "rebounds_lower", "rebounds_upper", "rebounds_median",
        "rebounds_std_pred", "rebounds_std_epistemic", "rebounds_std_aleatoric",
        "rebounds_std80_lower", "rebounds_std80_upper", "rebounds_pi80_width",
    
        "assists_mean", "assists_lower", "assists_upper", "assists_median",
        "assists_std_pred", "assists_std_epistemic", "assists_std_aleatoric",
        "assists_std80_lower", "assists_std80_upper", "assists_pi80_width",
    
        "steals_mean", "steals_lower", "steals_upper", "steals_median",
        "steals_std_pred", "steals_std_epistemic", "steals_std_aleatoric",
        "steals_std80_lower", "steals_std80_upper", "steals_pi80_width",
    
        "blocks_mean", "blocks_lower", "blocks_upper", "blocks_median",
        "blocks_std_pred", "blocks_std_epistemic", "blocks_std_aleatoric",
        "blocks_std80_lower", "blocks_std80_upper", "blocks_pi80_width",
    
        "points_mean", "points_lower", "points_upper", "points_median",
        "points_std_pred", "points_std_epistemic", "points_std_aleatoric",
        "points_std80_lower", "points_std80_upper", "points_pi80_width",
    ]
    
    # hard stop if no rows
    if out.empty:
        print("ERROR: No rows in `out` — aborting save.")
        sys.exit(1)
    
    # hard stop if any required column is missing
    missing = [c for c in cols_to_keep if c not in out.columns]
    if missing:
        print("ERROR: Missing required columns:", missing)
        print("Present columns:", list(out.columns))
        sys.exit(1)
    
    # only keep required columns
    out = out[cols_to_keep].copy()
    
    # --- Clamp nonnegative stat summaries for DISPLAY-ONLY parquet ---
    # Don't touch std/variance columns except for *_std80_lower (a "bound-like" column)
    nonneg_prefixes = [
        "three_point_field_goals_made",  # 3PM
        "rebounds", "assists", "steals", "blocks", "points",
    ]
    summary_suffixes = ["mean", "median", "lower", "upper"]
    
    # 1) Clamp summary stats to [0, inf)
    for p in nonneg_prefixes:
        for s in summary_suffixes:
            col = f"{p}_{s}"
            if col in out.columns:
                out[col] = out[col].clip(lower=0.0)
    
    # 2) Clamp the std-derived lower bound used for an ~80% interval
    for p in nonneg_prefixes:
        low = f"{p}_std80_lower"
        if low in out.columns:
            out[low] = out[low].clip(lower=0.0)
    
    # 3) Recompute PI80 width from the (now clamped) bounds so the table is consistent
    for p in nonneg_prefixes:
        lo, hi, width = f"{p}_lower", f"{p}_upper", f"{p}_pi80_width"
        if lo in out.columns and hi in out.columns and width in out.columns:
            lo_c = out[lo].clip(lower=0.0)
            hi_c = out[hi].clip(lower=0.0)
            out[lo] = lo_c
            out[hi] = hi_c
            out[width] = (hi_c - lo_c).clip(lower=0.0)
        
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)  # no-op if it already exists
    today_str   = date.today().strftime("%Y-%m-%d")  # ISO with dashes
    output_path = f"{PREDICTIONS_DIR}/nba_predictions_{today_str}.parquet"
    out.to_parquet(output_path, index=False)  # overwrites if present
    out.to_parquet('C:/Users/jgmot/OneDrive - Southern Methodist University/Documents/GitHub/NBA_Stat_Pred/predictions.parquet', index=False)
    print(f"Saved {output_path} (rows={len(out)})")
    print("Saved predictions.parquet to local GitHub Repository folder.")
    
    # ---- auto-commit & push to GitHub ----
    REPO_DIR   = Path(r"C:\Users\jgmot\OneDrive - Southern Methodist University\Documents\GitHub\NBA_Stat_Pred")
    BRANCH     = "main"  # change if your default is 'master'
    COMMIT_MSG = f"predictions: {datetime.date.today().isoformat()}"
    
    def run(cmd):
        return subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=True)
    
    try:
        status = run(["git", "status", "--porcelain"])
        if status.stdout.strip():
            run(["git", "add", "-A"])
            # commit only if something is staged
            diff_cached = run(["git", "diff", "--cached", "--quiet"])
            if diff_cached.returncode != 0:  # 1 = there are staged changes
                c = run(["git", "commit", "-m", COMMIT_MSG])
                if c.returncode != 0:
                    print("Git commit failed:\n", c.stderr)
            p = run(["git", "push", "origin", BRANCH])
            if p.returncode != 0:
                print("Git push failed:\n", p.stderr)
            else:
                # NEW: success message
                head = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
                print(f"Git push successful → origin/{BRANCH} (HEAD {head})")
        else:
            print("No repo changes; nothing to commit.")
    except FileNotFoundError:
        print("Git not found on PATH. Install Git or use full path to git.exe (e.g., r'C:\\Program Files\\Git\\cmd\\git.exe').")

if __name__ == "__main__":
    main()
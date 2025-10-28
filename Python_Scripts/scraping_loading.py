# -*- coding: utf-8 -*-
# scraping_loading.py
"""
Created on Sat Sep 27 2025
@author: jgmot
"""

from __future__ import annotations
import os
import re
import time
import random
import types
import warnings
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoreadvancedv2, leaguegamelog

warnings.filterwarnings("ignore")

# -----------------------------
# Config (defaults match your script)
# -----------------------------
DATASETS_DIR = "datasets"
GAME_METADATA_PARQUET = f"{DATASETS_DIR}/game_metadata.parquet"
SEASON_CHUNKS_DIR = f"{DATASETS_DIR}/season_parquet_chunks"
ALL_SEASONS_PARQUET = f"{DATASETS_DIR}/nba_advanced_boxscores_all_seasons.parquet"
MERGED_ADV_PARQUET = f"{DATASETS_DIR}/nba_advanced_boxscores_with_metadata.parquet"

# Renamed output for the joined/cleaned gamelogs, ready for feature engineering
READY_FOR_FE_PATH = f"{DATASETS_DIR}/gamelogs_ready_for_fe.parquet"

# Backward-compat alias (do NOT advertise; here to avoid breaks)
PREPROCESSED_PATH = READY_FOR_FE_PATH

GAMELOGS_PATH = f"{DATASETS_DIR}/nba_gamelogs.parquet"
LINEUPS_PATH = f"{DATASETS_DIR}/lineup_stints.parquet"
ONOFF_PATH = f"{DATASETS_DIR}/onoff_player_game.parquet"
INJURY_A = f"{DATASETS_DIR}/Injury Database.csv"
INJURY_B = f"{DATASETS_DIR}/2025_injuries.csv"
INJURY_C = f"{DATASETS_DIR}/current_injuries.parquet"

CHUNK_SIZE = 120
CHUNK_SLEEP = 420.0  # 7 minutes
PER_REQUEST_SLEEP = (2.0, 4.0)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------ training constants carried through ------------------
targets = ['three_point_field_goals_made', 'rebounds', 'assists', 'steals', 'blocks', 'points']
targets2 = ['minutes', 'field_goals_attempted', 'field_goals_made',
            'free_throws_attempted', 'free_throws_made',
            'three_point_field_goals_attempted'] + targets


# -----------------------------
# Helpers
# -----------------------------
def normalize_name(name: str) -> str:
    name = str(name).strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    # strip accents -> ascii
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    # kill punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    # collapse whitespace
    name = re.sub(r'\s+', ' ', name)
    # drop trailing roman numerals (II, III, IV, etc.)
    name = re.sub(r'\b(?:[IVX]+)$', '', name).strip()
    return name


def del_except(*keep):
    """Delete big objects from globals; keep only what's asked (mirrors your cleanup helper)."""
    deletable_types = (
        pd.DataFrame, pd.Series, pd.Index, np.ndarray,
        list, dict, set, tuple, int, float, str
    )
    keep = set(keep) | {'del_except'}
    for name, val in list(globals().items()):
        if name.startswith('_') or name in keep:
            continue
        if isinstance(val, types.ModuleType):
            continue
        if isinstance(val, (types.FunctionType, type)):
            continue
        if isinstance(val, deletable_types):
            try:
                del globals()[name]
            except Exception:
                pass


# -----------------------------
# Pipeline class
# -----------------------------
class ScrapeLoadPipeline:
    def __init__(
        self,
        datasets_dir: str = DATASETS_DIR,
        game_metadata_parquet: str = GAME_METADATA_PARQUET,
        season_chunks_dir: str = SEASON_CHUNKS_DIR,
        all_seasons_parquet: str = ALL_SEASONS_PARQUET,
        merged_adv_parquet: str = MERGED_ADV_PARQUET,
        ready_for_fe_path: str = READY_FOR_FE_PATH,
        # deprecated compat param (if someone still passes preprocessed_path=)
        preprocessed_path: Optional[str] = None,
        gamelogs_path: str = GAMELOGS_PATH,
        lineups_path: str = LINEUPS_PATH,
        onoff_path: str = ONOFF_PATH,
        injury_a: str = INJURY_A,
        injury_b: str = INJURY_B,
        injury_c: str = INJURY_C,
        chunk_size: int = CHUNK_SIZE,
        chunk_sleep: float = CHUNK_SLEEP,
        per_request_sleep: Tuple[float, float] = PER_REQUEST_SLEEP,
    ):
        self.datasets_dir = datasets_dir
        self.game_metadata_parquet = game_metadata_parquet
        self.season_chunks_dir = season_chunks_dir
        self.all_seasons_parquet = all_seasons_parquet
        self.merged_adv_parquet = merged_adv_parquet

        # prefer the new name; if someone passed preprocessed_path explicitly, honor it
        self.ready_for_fe_path = preprocessed_path or ready_for_fe_path

        self.gamelogs_path = gamelogs_path
        self.lineups_path = lineups_path
        self.onoff_path = onoff_path
        self.injury_a = injury_a
        self.injury_b = injury_b
        self.injury_c = injury_c

        self.chunk_size = chunk_size
        self.chunk_sleep = chunk_sleep
        self.per_request_sleep = per_request_sleep

        Path(self.datasets_dir).mkdir(parents=True, exist_ok=True)
        Path(self.season_chunks_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Stage 1: scrape metadata ----------
    def get_game_metadata(self, seasons: Iterable[str], delay: float = 1.0) -> Optional[pd.DataFrame]:
        all_logs: List[pd.DataFrame] = []
        for season in seasons:
            for season_type in ["Regular Season", "Playoffs"]:
                try:
                    time.sleep(delay)  # prevent API throttling
                    log_df = leaguegamelog.LeagueGameLog(
                        season=season,
                        season_type_all_star=season_type,
                        timeout=10
                    ).get_data_frames()[0]
                    log_df = (
                        log_df[['GAME_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP']]
                        .drop_duplicates()
                        .copy()
                    )
                    log_df['SEASON'] = season
                    log_df['SEASON_TYPE'] = season_type
                    all_logs.append(log_df)
                except Exception as e:
                    print(f"Failed to load {season} ({season_type}): {e}")

        if not all_logs:
            print("No metadata retrieved.")
            return None

        metadata_df = pd.concat(all_logs, ignore_index=True)
        metadata_df.to_parquet(self.game_metadata_parquet, index=False)
        print(f"Game metadata saved to: {self.game_metadata_parquet}")
        return metadata_df

    # ---------- Stage 2: per-game advanced box ----------
    def _get_advanced_boxscore(self, game_id: str, retries: int = 2, timeout: int = 10, backoff: float = 2.5) -> Optional[pd.DataFrame]:
        attempt = 0
        while attempt < retries:
            try:
                time.sleep(random.uniform(*self.per_request_sleep))
                box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id, timeout=timeout)
                df = box.get_data_frames()[0]
                df['GAME_ID'] = game_id
                return df
            except Exception as e:
                print(f"Error for {game_id} on attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(backoff * attempt)
        return None

    def process_season_in_chunks(self, season: str, metadata_df: pd.DataFrame, overwrite: bool = False) -> Optional[pd.DataFrame]:
        outfile = os.path.join(self.season_chunks_dir, f"advanced_boxscore_{season.replace('-', '_')}.parquet")
        existing_df = pd.read_parquet(outfile) if (os.path.exists(outfile) and not overwrite) else pd.DataFrame()

        all_game_ids = metadata_df.loc[metadata_df["SEASON"] == season, "GAME_ID"].drop_duplicates().astype(str).tolist()
        already_done = set(existing_df["GAME_ID"].astype(str)) if not existing_df.empty else set()
        remaining_ids = [gid for gid in all_game_ids if gid not in already_done]

        print(f"Processing {len(remaining_ids)} remaining games for {season}...")
        collected = [existing_df] if not existing_df.empty else []

        for i in range(0, len(remaining_ids), self.chunk_size):
            chunk = remaining_ids[i:i + self.chunk_size]
            chunk_data: List[pd.DataFrame] = []
            print(f"Starting chunk {i // self.chunk_size + 1} of {season}...")

            for game_id in tqdm(chunk, desc=f"{season} - Chunk {i // self.chunk_size + 1}"):
                df = self._get_advanced_boxscore(game_id)
                if df is not None:
                    df["SEASON"] = season
                    chunk_data.append(df)

            if chunk_data:
                chunk_df = pd.concat(chunk_data, ignore_index=True)
                collected.append(chunk_df)
                combined_df = pd.concat(collected, ignore_index=True)
                combined_df.to_parquet(outfile, index=False)
                print(f"Chunk saved to {outfile}")

            # cooldown if more chunks remain
            if self.chunk_sleep and (i + self.chunk_size < len(remaining_ids)):
                print(f"Sleeping for {self.chunk_sleep / 60:.1f} minutes to avoid throttling...")
                time.sleep(self.chunk_sleep)

        if not os.path.exists(outfile):
            print(f"No chunk data saved for {season}.")
            return None

        final_df = pd.read_parquet(outfile)
        print(f"Finished season {season}: {final_df['GAME_ID'].nunique()} games saved.")
        return final_df

    def scrape_all_seasons(self, seasons: Iterable[str], metadata_df: pd.DataFrame, overwrite: bool = False) -> List[pd.DataFrame]:
        per_season: List[pd.DataFrame] = []
        for season in seasons:
            df = self.process_season_in_chunks(season, metadata_df=metadata_df, overwrite=overwrite)
            if df is not None:
                per_season.append(df)
        return per_season

    def combine_and_merge(self, per_season_dfs: List[pd.DataFrame], metadata_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if not per_season_dfs:
            print("No data collected.")
            return None, None

        combined_df = pd.concat(per_season_dfs, ignore_index=True)
        combined_df.to_parquet(self.all_seasons_parquet, index=False)
        print(f"All seasons combined and saved to: {self.all_seasons_parquet}")

        merged_df = combined_df.merge(
            metadata_df,
            on=["GAME_ID", "TEAM_ABBREVIATION"],
            how="left"
        )
        merged_df.to_parquet(self.merged_adv_parquet, index=False)
        print(f"Merged data saved to {self.merged_adv_parquet}")
        return combined_df, merged_df
    
    # ---------- Injury helpers (A/B/C unified) ----------
    @staticmethod
    def _prep_injury_ab(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize A/B CSVs:
          - PLAYER: 'Achiuwa, Precious' -> 'Precious Achiuwa' (via normalize_name)
          - TEAM:   strip (keep case as in your source)
          - DATE:   to datetime, normalized to midnight (no tz)
          - Keep STATUS/REASON if present
        Output columns: TEAM, DATE, PLAYER, norm, STATUS?, REASON?
        """
        out = df.copy()
    
        # Ensure expected columns exist
        for col in ("PLAYER", "TEAM", "DATE"):
            if col not in out.columns:
                out[col] = pd.NA
    
        # Normalize player name -> "norm"
        out["PLAYER"] = out["PLAYER"].astype(str).str.strip()
        out["norm"] = out["PLAYER"].apply(normalize_name)
    
        # Normalize team (strip only to match your join)
        out["TEAM"] = out["TEAM"].astype(str).str.strip()
    
        # Normalize date (CSV looks like 10/22/2024)
        out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce").dt.normalize()
    
        # Keep the common columns and optional status fields
        keep = ["TEAM", "DATE", "PLAYER", "norm"]
        for c in ("STATUS", "REASON"):
            if c in out.columns:
                out[c] = out[c].astype(str).str.strip()
                keep.append(c)
    
        out = out[keep]
    
        # Drop rows where any key is missing
        out = out.dropna(subset=["TEAM", "DATE", "norm"])
    
        # Deduplicate exact rows across all kept columns
        out = out.drop_duplicates().reset_index(drop=True)
        return out
    
    @staticmethod
    def _parse_injury_c_date(x) -> pd.Timestamp | None:
        """INJURY_C.updated like 'Tue, Sep 23, 2025' -> pandas Timestamp date."""
        if pd.isna(x):
            return None
        s = str(x).strip()
        for fmt in ("%a, %b %d, %Y", "%A, %b %d, %Y", "%b %d, %Y"):
            try:
                return pd.to_datetime(pd.to_datetime(s, format=fmt)).normalize()
            except Exception:
                pass
        try:
            return pd.to_datetime(s, errors="coerce").normalize()
        except Exception:
            return None
    
    @staticmethod
    def _prep_injury_c(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize current_injuries.parquet:
          - player: may include punctuation/accents (e.g., 'T.J. McConnell')
          - team:   strip only (to match your join)
          - updated: e.g., 'Tue, Sep 30, 2025' -> DATE normalized to midnight
          - STATUS/REASON parsed from description: 'Out (Achilles) - ...'
        Output columns: TEAM, DATE, PLAYER, norm, STATUS, REASON
        """
        out = df.rename(columns={
            "player": "PLAYER",
            "team": "TEAM",
            "updated": "UPDATED",
            "description": "DESCRIPTION",
        }).copy()
    
        # DATE from UPDATED (robust parser already defined)
        out["DATE"] = out["UPDATED"].apply(ScrapeLoadPipeline._parse_injury_c_date)
    
        # Parse STATUS/REASON from DESCRIPTION (e.g., 'Out (Achilles) - ...')
        def _status_reason(s: str) -> tuple[str | None, str | None]:
            if pd.isna(s) or not str(s).strip():
                return (None, None)
            txt = str(s).strip()
            m = re.match(r"^\s*([A-Za-z][A-Za-z\- ]+?)\s*(?:\(([^)]+)\))?", txt)
            status = m.group(1).strip().title() if m else None
            reason = m.group(2).strip() if (m and m.group(2)) else None
            canon = {
                "Out","Doubtful","Questionable","Probable","Available",
                "Inactive","Day-To-Day","Two-Way","Ineligible",
                "Return To Competition Reconditioning"
            }
            if status:
                for w in canon:
                    if status.lower().startswith(w.lower()):
                        status = w
                        break
            return (status, reason)
    
        sr = out["DESCRIPTION"].apply(_status_reason)
        out["STATUS"] = [t[0] for t in sr]
        out["REASON"] = [t[1] for t in sr]
    
        # Normalize player & team
        out["PLAYER"] = out["PLAYER"].astype(str).str.strip()
        out["norm"] = out["PLAYER"].apply(normalize_name)
        out["TEAM"] = out["TEAM"].astype(str).str.strip()
    
        keep = ["TEAM", "DATE", "PLAYER", "norm", "STATUS", "REASON"]
        out = out[keep]
    
        # Drop rows where any key is missing
        out = out.dropna(subset=["TEAM", "DATE", "norm"])
    
        # Deduplicate exact rows across all kept columns
        out = out.drop_duplicates().reset_index(drop=True)
        return out

    # ---------- Stage 3: load R data, injuries, on/off, lineups; build ready-for-FE ----------
    def build_ready_for_fe(self) -> pd.DataFrame:
        """
        Builds `gamelogs_ready_for_fe.parquet` (ready for feature engineering).
        Uses cache only if the source gamelogs contain no new game_ids.
        """
    
        if os.path.exists(self.ready_for_fe_path):
            try:
                cached = pd.read_parquet(self.ready_for_fe_path)
                src    = pd.read_parquet(self.gamelogs_path)
                adv    = pd.read_parquet(self.merged_adv_parquet) 

                def _resolve_idcol(df, preferred):
                    if preferred in df.columns: return preferred
                    low = preferred.lower()
                    return low if low in df.columns else None

                ALL_STAR = {"DUR","LEB","GIA","WEST","EAST","CAN","CHK","KEN","SHQ"}

                # NEW: use global targets2 if present; otherwise fallback
                _targets2 = set(targets2) if "targets2" in globals() else {
                    "minutes","field_goals_attempted","field_goals_made",
                    "free_throws_attempted","free_throws_made",
                    "three_point_field_goals_attempted",
                    "three_point_field_goals_made","rebounds","assists",
                    "steals","blocks","points"
                }

                # Normalize frames (lowercase column names to match assembly)
                c = cached.copy(); c.columns = c.columns.str.lower()
                s = src.copy();    s.columns = s.columns.str.lower()
                a = adv.copy()  # NEW

                # Same prefilters as assembly on source gamelogs
                if "team_abbreviation" in s.columns:
                    s = s[~s["team_abbreviation"].isin(ALL_STAR)]
                if "game_date" in s.columns:
                    s["game_date"] = pd.to_datetime(s["game_date"], errors="coerce")
                    s = s[s["game_date"].dt.year >= 2021]
                if "did_not_play" in s.columns:
                    s = s[s["did_not_play"] == False]
                if "minutes" in s.columns:
                    s = s[s["minutes"] > 0]
                for col in (_targets2 & set(s.columns)):
                    s = s[s[col].notna()]

                # mirror the advanced-join gate used in assembly
                a["GAME_DATE"] = pd.to_datetime(a["GAME_DATE"], errors="coerce")
                a["norm"] = a["PLAYER_NAME"].astype(str).apply(normalize_name) 
                s["norm"] = s["athlete_display_name"].astype(str).apply(normalize_name)
                adv_keys = (                      
                    a[["GAME_DATE","TEAM_ABBREVIATION","norm"]]
                    .dropna()
                    .drop_duplicates()
                    .rename(columns={"GAME_DATE":"game_date","TEAM_ABBREVIATION":"team_abbreviation"})
                )
                s = s.merge(adv_keys, on=["game_date","team_abbreviation","norm"], how="left")  

                # NEW: enforce the same player/season constraints as assembly
                if {"athlete_display_name","season","minutes"}.issubset(s.columns) and len(s):
                    cnt = s.groupby("athlete_display_name").size()           
                    s = s[s["athlete_display_name"].isin(cnt[cnt >= 10].index)]   

                    qual = (                                                          
                        s.groupby(["athlete_display_name","season"])["minutes"]
                         .mean().reset_index()
                         .query("minutes >= 15")[["athlete_display_name","season"]]
                    )
                    s = s[s[["athlete_display_name","season"]]
                          .apply(tuple, axis=1)
                          .isin(qual.apply(tuple, axis=1))]                             

                    recent = s["season"].max()                 
                    s = s[s["season"] == recent]                                            

                # Resolve IDs after all normalization/prefilters
                c_id = _resolve_idcol(c, "GAME_ID")
                s_id = _resolve_idcol(s, "GAME_ID")
                if not c_id or not s_id:
                    print("Could not resolve GAME_ID/game_id in cached/src after normalization; rebuilding ready-for-FE parquet.")
                else:
                    cached_ids = pd.Index(c[c_id].astype(str).unique())
                    src_ids    = pd.Index(s[s_id].astype(str).unique())
                    new_ids    = src_ids.difference(cached_ids)

                    cmax = pd.to_datetime(c.get("game_date"), errors="coerce").max()
                    smax = pd.to_datetime(s.get("game_date"), errors="coerce").max()
                    newer_dates = (pd.notna(smax) and pd.notna(cmax) and (smax > cmax))

                    if len(new_ids) == 0 and not newer_dates:
                        print("Using cached gamelogs (assembly stage)")
                        return cached
                    else:
                        print(f"Detected {len(new_ids)} new assembly-relevant game_id(s) or newer dates; rebuilding ready-for-FE parquet.")  # CHANGED: clearer message
            except Exception as e:
                print(f"Cache check failed ({e}); rebuilding ready-for-FE parquet.")

        print("Assembling gamelogs")

        # Read merged advanced boxscores
        combined_df_merged = pd.read_parquet(self.merged_adv_parquet)
        if "GAME_DATE" in combined_df_merged.columns:
            combined_df_merged["GAME_DATE"] = pd.to_datetime(
                combined_df_merged["GAME_DATE"], errors="coerce"
            )
        combined_df_merged = combined_df_merged.dropna().reset_index(drop=True)

        # Inputs (from R + others)
        A = pd.read_csv(self.injury_a)
        B = pd.read_csv(self.injury_b)
        inj_frames = [self._prep_injury_ab(A), self._prep_injury_ab(B)]

        # Current injuries from Basketball-Reference (parquet)
        if not self.injury_c:
            raise ValueError("injury_c path is not set. Provide a parquet path for current injuries.")
        if not os.path.exists(self.injury_c):
            raise FileNotFoundError(
                f"Required file not found: {self.injury_c}. "
                "Export or place the Basketball-Reference current-injuries parquet there."
            )
        C_raw = pd.read_parquet(self.injury_c)
        inj_frames.append(self._prep_injury_c(C_raw))

        injury_db = pd.concat(inj_frames, ignore_index=True)
        
        # Standardize keys again defensively
        injury_db["TEAM"] = injury_db["TEAM"].astype(str).str.strip()
        injury_db["norm"] = injury_db["norm"].astype(str).str.strip()
        injury_db["DATE"] = pd.to_datetime(injury_db["DATE"], errors="coerce").dt.normalize()
        
        # Keep only rows where STATUS is Out* (if STATUS missing, treat as NA and drop below)
        if "STATUS" not in injury_db.columns:
            injury_db["STATUS"] = pd.NA
        injury_db["STATUS"] = injury_db["STATUS"].astype("string").str.strip()
        injury_db = injury_db[injury_db["STATUS"].str.lower().str.startswith("out", na=False)].copy()
        
        # Drop rows with missing keys
        injury_db = injury_db.dropna(subset=["TEAM", "DATE", "norm"])
        
        # 1) Remove exact duplicates across all columns we carry
        injury_db = injury_db.drop_duplicates().reset_index(drop=True)
        
        # 2) Keep only one row per (TEAM, DATE, norm)
        injury_db = (
            injury_db
            .sort_values(["DATE", "TEAM", "norm"])
            .drop_duplicates(subset=["TEAM", "DATE", "norm"], keep="first")
            .reset_index(drop=True)
        )

        gamelogs = pd.read_parquet(self.gamelogs_path)
        
        if "game_date" in gamelogs.columns:
            gamelogs["game_date"] = pd.to_datetime(gamelogs["game_date"], errors="coerce")
            
        lineups  = pd.read_parquet(self.lineups_path)
        onoff    = pd.read_parquet(self.onoff_path)
        
        # Basic cleanup
        gamelogs = gamelogs.dropna().reset_index(drop=True)
        if len(gamelogs.select_dtypes('float64').columns):
            gamelogs[gamelogs.select_dtypes('float64').columns] = (
                gamelogs.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
            )
        if len(gamelogs.select_dtypes('int64').columns):
            gamelogs[gamelogs.select_dtypes('int64').columns] = (
                gamelogs.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
            )

        # Post-2020 filter
        injury_db = injury_db[injury_db['DATE'].dt.year >= 2021].copy()
        gamelogs = gamelogs[gamelogs['game_date'].dt.year >= 2021].copy().reset_index(drop=True)

        # Starter injuries by team/date  (<<< INTEGRATED BLOCK START)
        if 'starter' not in gamelogs.columns:
            gamelogs['starter'] = False

        # Compute starter rate per player
        starter_rate = (
            gamelogs.groupby("athlete_display_name")["starter"]
                    .mean()          # fraction of games started
                    .reset_index()
        )

        # Keep only players who started >50% of their games
        regular_starters = set(starter_rate.loc[starter_rate["starter"] > 0.5, "athlete_display_name"])

        # Filter to those in your norm set (if you still need both)
        starter_norms = set(gamelogs.loc[
            gamelogs["athlete_display_name"].isin(regular_starters),
            "athlete_display_name"
        ])

        starter_injuries = injury_db[injury_db["norm"].isin(starter_norms)].copy()
        starter_injuries["starter_injured"] = True
        injured_keys = starter_injuries[["TEAM","DATE", "starter_injured"]].drop_duplicates()

        # Note: gamelogs uses 'team_display_name' to match TEAM, and 'game_date' to match DATE
        gamelogs = gamelogs.merge(
            injured_keys,
            left_on=["team_display_name","game_date"],
            right_on=["TEAM","DATE"],
            how="left",
        )

        # Ensure the flag exists
        gamelogs["starter_injured"] = gamelogs["starter_injured"].fillna(False)
        print("After merging injuries:", gamelogs.shape, "starter_injured True =", int(gamelogs["starter_injured"].sum()))
        # (<<< INTEGRATED BLOCK END)

        # CONSISTENT name normalization (for advanced join)
        gamelogs['norm'] = gamelogs['athlete_display_name'].apply(normalize_name)

        # ===========================================================
        # 6) JOIN ADVANCED BOXSCORES (adv) BY (game_date, team_abbreviation, norm)
        # ===========================================================
        adv = combined_df_merged.copy()
        if "GAME_DATE" in adv.columns:
            adv["GAME_DATE"] = pd.to_datetime(adv["GAME_DATE"], errors="coerce")
        adv["norm"] = adv["PLAYER_NAME"].astype(str).apply(normalize_name)

        gamelogs = gamelogs.merge(
            adv,
            left_on=["game_date","team_abbreviation","norm"],
            right_on=["GAME_DATE","TEAM_ABBREVIATION","norm"],
            how="left"
        )

        # Keep only rows that found a matching player in advanced
        gamelogs = gamelogs[gamelogs["norm"].isin(adv["norm"])].copy()
        print("After joining adv:", gamelogs.shape)

        # ===========================================================
        # 7) DROP HOUSEKEEPING COLUMNS (feel free to adjust list)
        # ===========================================================
        drop_cols = [
            'SEASON_x', 'SEASON_y', 'TEAM_ABBREVIATION', 'GAME_DATE', 'SEASON_TYPE',
            'norm', 'COMMENT', 'MIN', 'NICKNAME', 'START_POSITION', 'PLAYER_NAME',
            'PLAYER_ID', 'TEAM_CITY', 'TEAM_ID', 'GAME_ID', 'TEAM', 'DATE', 'MATCHUP',
            'team_display_name'
        ]
        gamelogs = gamelogs.drop(columns=drop_cols, errors='ignore')

        # ===========================================================
        # 8) FILTERS: completeness, DNPs, minutes>0, player volume, season quals
        # ===========================================================
        # Completeness for targets2
        gamelogs = gamelogs.dropna(subset=[c for c in targets2 if c in gamelogs.columns]).copy()

        # Remove players with < 10 games (after filters so far)
        player_game_counts = gamelogs.groupby("athlete_display_name").size()
        valid_players = player_game_counts[player_game_counts >= 10].index
        gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

        # DNP / minutes>0
        if "did_not_play" in gamelogs.columns:
            gamelogs = gamelogs[gamelogs["did_not_play"] == False]
        if "minutes" in gamelogs.columns:
            gamelogs = gamelogs[gamelogs["minutes"] > 0]
        gamelogs = gamelogs.reset_index(drop=True)

        # Sort to stabilize future rolling features
        sort_cols = [c for c in ['athlete_display_name','game_date','season','team_abbreviation'] if c in gamelogs.columns]
        if sort_cols:
            gamelogs = gamelogs.sort_values(sort_cols).reset_index(drop=True)

        # Season qualification: mean minutes >= 15 in a season
        if {"athlete_display_name","season","minutes"}.issubset(gamelogs.columns):
            qual = (
                gamelogs.groupby(['athlete_display_name','season'])['minutes']
                        .mean().reset_index()
                        .query('minutes >= 15')[['athlete_display_name','season']]
            )
            keep_pairs = set(map(tuple, qual.values))
            mask = gamelogs[['athlete_display_name','season']].apply(tuple, axis=1).isin(keep_pairs)
            gamelogs = gamelogs[mask].copy().reset_index(drop=True)

        # Most recently completed season players only
        if "season" in gamelogs.columns:
            recent_season = 2025
            cur_players = gamelogs[gamelogs['season'] == recent_season]['athlete_display_name'].unique()
            gamelogs = gamelogs[gamelogs['athlete_display_name'].isin(cur_players)].copy().reset_index(drop=True)

        print("After filters:", gamelogs.shape)

        # ===========================================================
        # 9) LINEUP STATS + ON/OFF MERGES
        # ===========================================================
        # Prepare lineup features
        lineups_local = lineups.copy()
        lineups_local['player_ids'] = lineups_local['lineup'].str.split('-')
        exploded = lineups_local.explode('player_ids').copy()
        exploded['player_id'] = exploded['player_ids'].astype('int64', errors='ignore')
        exploded['duration_minutes'] = pd.to_numeric(exploded['duration_minutes'], errors='coerce')
        exploded['points'] = pd.to_numeric(exploded['points'], errors='coerce')
        exploded['points_per_min'] = exploded['points'] / exploded['duration_minutes'].replace(0, np.nan)

        agg_stats = (
            exploded.groupby(['player_id','game_id'])
            .agg(total_lineup_minutes=('duration_minutes','sum'),
                 num_unique_lineups=('lineup','nunique'))
            .reset_index()
        )

        weighted_avg = (
            exploded.groupby(['player_id','game_id'])
            .apply(lambda df: np.average(df['points'], weights=df['duration_minutes']) if df['duration_minutes'].fillna(0).sum() else np.nan)
            .reset_index(name='avg_lineup_ppm')
        )

        player_lineup_features = pd.merge(agg_stats, weighted_avg, on=['player_id','game_id'], how='outer')

        # On/Off cleanup and dedupe
        onoff_local = onoff.copy()
        onoff_local['player_id'] = pd.to_numeric(onoff_local['player_id'], errors='coerce').astype('Int64')
        onoff_local = onoff_local.drop_duplicates(subset=['player_id','game_id'])

        # Merge into gamelogs
        gamelogs = gamelogs.merge(onoff_local, on=['player_id','game_id'], how='left')
        gamelogs = gamelogs.merge(player_lineup_features, on=['player_id','game_id'], how='left')

        # cache
        Path(self.ready_for_fe_path).parent.mkdir(parents=True, exist_ok=True)
        gamelogs.to_parquet(self.ready_for_fe_path, index=False)
        print("Saved gamelogs_ready_for_fe.parquet")
        return gamelogs

    # Backward-compat method (delegates to the new name)
    def build_preprocessed(self) -> pd.DataFrame:
        return self.build_ready_for_fe()

    # ---------- Full scrape-or-load orchestrator (optional) ----------
    def run_full_scrape_then_build(
        self,
        seasons: Iterable[str],
        delay: float = 1.0,
        overwrite_chunks: bool = False
    ) -> pd.DataFrame:
        """If you want to trigger scraping too; else call build_ready_for_fe directly when parquets exist."""
        metadata_df = self.get_game_metadata(seasons, delay=delay)
        if metadata_df is None:
            raise RuntimeError("No metadata; cannot proceed.")

        per_season = self.scrape_all_seasons(seasons, metadata_df, overwrite=overwrite_chunks)
        _, _ = self.combine_and_merge(per_season, metadata_df)

        return self.build_ready_for_fe()


# -----------------------------
# Public convenience function
# -----------------------------
def run_scrape_and_load(
    seasons: Iterable[str] = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26"),
    *,
    ensure_scrape: bool = False,
    delay: float = 1.0,
    overwrite_chunks: bool = False,
    datasets_dir: str = DATASETS_DIR,
    game_metadata_parquet: str = GAME_METADATA_PARQUET,
    season_chunks_dir: str = SEASON_CHUNKS_DIR,
    all_seasons_parquet: str = ALL_SEASONS_PARQUET,
    merged_adv_parquet: str = MERGED_ADV_PARQUET,
    # preferred new kwarg
    ready_for_fe_path: str = READY_FOR_FE_PATH,
    # deprecated compat kwarg; if provided, it overrides ready_for_fe_path
    preprocessed_path: Optional[str] = None,
    gamelogs_path: str = GAMELOGS_PATH,
    lineups_path: str = LINEUPS_PATH,
    onoff_path: str = ONOFF_PATH,
    injury_a: str = INJURY_A,
    injury_b: str = INJURY_B,
    injury_c: str = INJURY_C,
) -> pd.DataFrame:
    """
    Primary entry point.

    - If `ensure_scrape=True`, it will scrape metadata and advanced boxscores first,
      combine/merge, then build the ready-for-FE parquet.
    - If `ensure_scrape=False` (default), it will assume your parquet inputs exist and
      only (re)build the ready-for-FE parquet if missing.
    """
    pipe = ScrapeLoadPipeline(
        datasets_dir=datasets_dir,
        game_metadata_parquet=game_metadata_parquet,
        season_chunks_dir=season_chunks_dir,
        all_seasons_parquet=all_seasons_parquet,
        merged_adv_parquet=merged_adv_parquet,
        ready_for_fe_path=(preprocessed_path or ready_for_fe_path),
        gamelogs_path=gamelogs_path,
        lineups_path=lineups_path,
        onoff_path=onoff_path,
        injury_a=injury_a,
        injury_b=injury_b,
        injury_c=injury_c,
    )
    
    # ===============================================================
    # 🧠 Guard: check whether the newest season (2025-26) is live yet
    # ===============================================================
    if ensure_scrape:
        try:
            # Quick probe: try to pull the LeagueGameLog for 2025-26
            from nba_api.stats.endpoints import leaguegamelog
            probe = leaguegamelog.LeagueGameLog(season="2025-26", season_type_all_star="Regular Season").get_data_frames()[0]
            if probe.empty:
                print("⚠️  NBA API returned no data for 2025-26 yet. Falling back to cached mode (no scrape).")
                ensure_scrape = False
            else:
                # sanity check: see if it contains a game within the past week
                latest_date = pd.to_datetime(probe["GAME_DATE"], errors="coerce").max()
                if pd.isna(latest_date) or latest_date < pd.Timestamp("2025-10-18"):
                    print("⚠️  Detected no recent 2025-26 games in NBA API. Falling back to cached mode.")
                    ensure_scrape = False
                else:
                    print(f"✅ Detected NBA API data for 2025-26 (latest game date: {latest_date.date()}). Proceeding with scrape.")
        except Exception as e:
            print(f"⚠️  NBA API probe failed ({e}). Falling back to cached mode.")
            ensure_scrape = False
    
    # ---------------------------------------------------------------
    # Continue with either scrape or cached mode depending on result
    # ---------------------------------------------------------------
    if ensure_scrape:
        return pipe.run_full_scrape_then_build(seasons, delay=delay, overwrite_chunks=overwrite_chunks)
    
    if not os.path.exists(merged_adv_parquet):
        raise FileNotFoundError(
            f"Missing {merged_adv_parquet}. "
            f"Set ensure_scrape=True to scrape/merge first or generate it upstream."
        )
    
    return pipe.build_ready_for_fe()

    if ensure_scrape:
        return pipe.run_full_scrape_then_build(seasons, delay=delay, overwrite_chunks=overwrite_chunks)

    # Only build the ready-for-FE parquet (no scraping); matches your "load from R then merge" path.
    if not os.path.exists(merged_adv_parquet):
        raise FileNotFoundError(
            f"Missing {merged_adv_parquet}. "
            f"Set ensure_scrape=True to scrape/merge first or generate it upstream."
        )
    return pipe.build_ready_for_fe()


# -----------------------------
# Quick manual test
# -----------------------------
if __name__ == "__main__":
    # Example usage:
    # 1) Only build if missing (assumes your parquets exist already)
    df = run_scrape_and_load(ensure_scrape=False)
    print(df.shape, "rows in gamelogs_ready_for_fe")

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
INJURY_A = "Injury Database.csv"
INJURY_B = "2025_injuries.csv"

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
    name = name.strip()
    if ',' in name:
        last, first = name.split(',', 1)
        name = f"{first.strip()} {last.strip()}"
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\b(?:[IVX]+)$', '', name)
    return name.strip()


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

    # ---------- Stage 3: load R data, injuries, on/off, lineups; build ready-for-FE ----------
    def build_ready_for_fe(self) -> pd.DataFrame:
        """Builds `gamelogs_ready_for_fe.parquet` (ready for feature engineering)."""
        if os.path.exists(self.ready_for_fe_path):
            print("Using cached gamelogs (assembly stage)")
            return pd.read_parquet(self.ready_for_fe_path)

        print("Assembling gamelogs")

        # Read merged advanced boxscores
        combined_df_merged = pd.read_parquet(self.merged_adv_parquet)
        combined_df_merged = combined_df_merged.dropna().reset_index(drop=True)

        # Inputs (from R + others)
        injury_db = pd.concat([
            pd.read_csv(self.injury_a),
            pd.read_csv(self.injury_b)
        ], ignore_index=True)

        gamelogs = pd.read_parquet(self.gamelogs_path)
        lineups = pd.read_parquet(self.lineups_path)
        onoff = pd.read_parquet(self.onoff_path)

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

        # Injury matching key
        injury_db['norm'] = injury_db['PLAYER'].apply(normalize_name)
        gamelogs['norm'] = (
            gamelogs['athlete_display_name']
              .str.replace(r'[^\w\s]', '', regex=True)
              .str.replace(r'\b(?:[IVX]+)$', '', regex=True)
              .str.strip()
        )

        # Dates
        injury_db['DATE'] = pd.to_datetime(injury_db['DATE'], errors='coerce')
        gamelogs['game_date'] = pd.to_datetime(gamelogs['game_date'], errors='coerce')
        combined_df_merged['GAME_DATE'] = pd.to_datetime(combined_df_merged['GAME_DATE'], errors='coerce')

        # Post-2020 filter
        injury_db = injury_db[injury_db['DATE'].dt.year >= 2021].copy()
        gamelogs = gamelogs[gamelogs['game_date'].dt.year >= 2021].copy().reset_index(drop=True)

        # Starter injuries by team/date
        if 'starter' not in gamelogs.columns:
            # If R export ever lacks it, default all False so pipeline doesn't break.
            gamelogs['starter'] = False

        starter_norms = set(gamelogs.loc[gamelogs['starter'] == True, 'norm'].unique())
        starter_injuries = injury_db[injury_db['norm'].isin(starter_norms)].copy()
        starter_injuries['starter_injured'] = True
        injured_team_dates = starter_injuries[['TEAM', 'DATE', 'starter_injured']].drop_duplicates(subset=['TEAM', 'DATE'])

        gamelogs = gamelogs.merge(
            injured_team_dates,
            left_on=['team_display_name', 'game_date'],
            right_on=['TEAM', 'DATE'],
            how='left'
        )
        # Ensure the column exists (no KeyError later)
        if 'starter_injured' not in gamelogs.columns:
            gamelogs['starter_injured'] = False
        gamelogs['starter_injured'] = gamelogs['starter_injured'].fillna(False)

        # Join advanced
        combined_df_merged['norm'] = combined_df_merged['PLAYER_NAME'].apply(normalize_name)
        gamelogs = gamelogs.merge(
            combined_df_merged,
            how='left',
            left_on=['game_date', 'team_abbreviation', 'norm'],
            right_on=['GAME_DATE', 'TEAM_ABBREVIATION', 'norm']
        )

        # keep only rows that matched a player in advanced
        gamelogs = gamelogs[gamelogs["norm"].isin(combined_df_merged["norm"])].copy()

        # drop housekeeping cols
        gamelogs = gamelogs.drop(columns=[
            'SEASON_x', 'SEASON_y', 'TEAM_ABBREVIATION', 'GAME_DATE', 'SEASON_TYPE',
            'norm', 'COMMENT', 'MIN', 'NICKNAME', 'START_POSITION', 'PLAYER_NAME',
            'PLAYER_ID', 'TEAM_CITY', 'TEAM_ID', 'GAME_ID', 'TEAM', 'DATE', 'MATCHUP',
            'team_display_name'
        ], errors='ignore')

        # target completeness + activity filters
        gamelogs = gamelogs.dropna(subset=targets2).copy()
        player_game_counts = gamelogs.groupby("athlete_display_name").size()
        valid_players = player_game_counts[player_game_counts >= 10].index
        gamelogs = gamelogs[gamelogs["athlete_display_name"].isin(valid_players)].copy()

        gamelogs = gamelogs[gamelogs["did_not_play"] == False].reset_index(drop=True)
        gamelogs = gamelogs[gamelogs["minutes"] > 0].reset_index(drop=True)

        # sort to stabilize cumulative features later
        gamelogs = gamelogs.sort_values(
            ['athlete_display_name', 'game_date', 'season', 'team_abbreviation']
        ).reset_index(drop=True)

        # season-qualification
        qualified_keys = (
            gamelogs
            .groupby(['athlete_display_name', 'season'])['minutes']
            .mean()
            .reset_index()
            .query('minutes >= 15')[['athlete_display_name', 'season']]
        )
        gamelogs = gamelogs[
            gamelogs[['athlete_display_name', 'season']].apply(tuple, axis=1).isin(
                qualified_keys.apply(tuple, axis=1)
            )
        ]

        # current season players only
        recent_season = gamelogs['season'].max()
        current_players = gamelogs[gamelogs['season'] == recent_season]['athlete_display_name'].unique()
        gamelogs = gamelogs[gamelogs['athlete_display_name'].isin(current_players)]

        # types / lowercase cols
        if len(gamelogs.select_dtypes('float64').columns):
            gamelogs[gamelogs.select_dtypes('float64').columns] = (
                gamelogs.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
            )
        if len(gamelogs.select_dtypes('int64').columns):
            gamelogs[gamelogs.select_dtypes('int64').columns] = (
                gamelogs.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
            )
        gamelogs.columns = gamelogs.columns.str.lower()

        # lineup-derived features
        lineups = lineups.copy()
        lineups['player_ids'] = lineups['lineup'].str.split('-')
        exploded = lineups.explode('player_ids').copy()
        exploded['player_id'] = exploded['player_ids'].astype(int)
        exploded['points_per_min'] = exploded['points'] / exploded['duration_minutes'].replace(0, np.nan)

        agg_stats = (
            exploded.groupby(['player_id', 'game_id'])
            .agg(total_lineup_minutes=('duration_minutes', 'sum'),
                 num_unique_lineups=('lineup', 'nunique'))
            .reset_index()
        )

        weighted_avg = (
            exploded.groupby(['player_id', 'game_id'])
            .apply(lambda df: np.average(df['points'], weights=df['duration_minutes']))
            .reset_index(name='avg_lineup_ppm')
        )

        player_lineup_features = pd.merge(agg_stats, weighted_avg, on=['player_id', 'game_id'])

        # on/off
        onoff = onoff.copy()
        onoff['player_id'] = onoff['player_id'].astype('int32')
        onoff = onoff.drop_duplicates(subset=['player_id', 'game_id'])

        # joins
        gamelogs = gamelogs.merge(onoff, on=['player_id', 'game_id'], how='left')
        gamelogs = gamelogs.merge(player_lineup_features, on=['player_id', 'game_id'], how='left')

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
    seasons: Iterable[str] = ("2021-22", "2022-23", "2023-24", "2024-25"),
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
    )

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

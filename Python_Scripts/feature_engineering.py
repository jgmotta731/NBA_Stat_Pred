# -*- coding: utf-8 -*-
#feature_engineering.py
"""
Created on Sat Sep 27 2025
@author: jgmot
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

DEFAULT_FEATURED_PATH = "datasets/gamelogs_features.parquet"

# Defaults used by your prior-features filter
DEFAULT_TARGETS = [
    "three_point_field_goals_made", "rebounds", "assists",
    "steals", "blocks", "points"
]
DEFAULT_TARGETS2 = [
    "minutes", "field_goals_attempted", "field_goals_made",
    "free_throws_attempted", "free_throws_made",
    "three_point_field_goals_attempted"
] + DEFAULT_TARGETS


def run_feature_engineering(
    gamelogs: pd.DataFrame,
    *,
    save_path: str = DEFAULT_FEATURED_PATH,
    cache_read: bool = True,
    cache_write: bool = True,
    targets: Optional[List[str]] = None,
    targets2: Optional[List[str]] = None,
    id_col: str = "GAME_ID",
    force_recompute: bool = False,
    ensure_fe: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:

    """
    Feature engineering stage.
    - Input: preprocessed `gamelogs` with columns used below.
    - Output: (engineered_gamelogs, feature_groups_dict)

    feature_groups_dict keys:
      - lagged_rolling_features
      - prior_features
      - numeric_features
      - categorical_features
      - embedding_features
      - features
      - targets
      - targets2
    """
    targets = targets or list(DEFAULT_TARGETS)
    targets2 = targets2 or list(DEFAULT_TARGETS2)
    
    # Treat ensure_fe as an alias for forcing recompute (same spirit as ensure_scrape)
    force_recompute = force_recompute or ensure_fe
    
    # ---------------- Cache short-circuit (HARD STOP if no new IDs) ----------------
    def _resolve_idcol(df: pd.DataFrame, preferred: str) -> Optional[str]:
        if preferred in df.columns:
            return preferred
        low = preferred.lower()
        return low if low in df.columns else None
    
    if force_recompute:
        print(f"Force-recompute requested → ignoring cached features at {save_path}")
    else:
        if cache_read and os.path.exists(save_path):
            fe_cached = pd.read_parquet(save_path)
            all_star_teams = ["DUR", "LEB", "GIA", "WEST", "EAST", "CAN", "CHK", "KEN", "SHQ"]
            in_id = _resolve_idcol(gamelogs, id_col)
            fe_id = _resolve_idcol(fe_cached, id_col)
    
            if in_id and fe_id:
                # 1) Apply the same early FE exclusion to the incoming frame
                g_in = gamelogs
                if "team_abbreviation" in g_in.columns:
                    g_in = g_in[~g_in["team_abbreviation"].isin(all_star_teams)]
    
                # 2) Compare IDs as strings to avoid dtype mismatches
                incoming_ids = pd.Index(g_in[in_id].astype(str).unique())
                cached_ids   = pd.Index(fe_cached[fe_id].astype(str).unique())
                new_ids = incoming_ids.difference(cached_ids)
    
                if len(new_ids) == 0:
                    # Detect updates to existing IDs (newer dates or increased row counts)
                    can_compare_dates = ("game_date" in g_in.columns) and ("game_date" in fe_cached.columns)
                    updates_due_to_dates = 0
                    updates_due_to_counts = 0
    
                    fe_cmp = fe_cached
                    if "team_abbreviation" in fe_cmp.columns:
                        fe_cmp = fe_cmp[~fe_cmp["team_abbreviation"].isin(all_star_teams)]
    
                    # Compare max game_date per ID
                    if can_compare_dates:
                        g_in_dates = pd.to_datetime(g_in["game_date"], errors="coerce")
                        fe_dates   = pd.to_datetime(fe_cmp["game_date"], errors="coerce")
                        in_latest = g_in.assign(_gd=g_in_dates).groupby(in_id)["_gd"].max()
                        fe_latest = fe_cmp.assign(_gd=fe_dates).groupby(fe_id)["_gd"].max()
                        fe_latest = fe_latest.reindex(in_latest.index)
                        newer_mask = in_latest.gt(fe_latest.fillna(pd.Timestamp(0)))
                        updates_due_to_dates = int(newer_mask.sum())
    
                    # Row count increases per ID (e.g., more player-rows)
                    in_counts = g_in.groupby(in_id).size()
                    fe_counts = fe_cmp.groupby(fe_id).size().reindex(in_counts.index).fillna(0).astype(int)
                    count_increase_mask = in_counts.gt(fe_counts)
                    updates_due_to_counts = int(count_increase_mask.sum())
    
                    if updates_due_to_dates > 0 or updates_due_to_counts > 0:
                        print(
                            f"Detected updates for {updates_due_to_dates} existing {in_id}(s) "
                            f"and count increases for {updates_due_to_counts} → recomputing features."
                        )
                    else:
                        print(f"No new {in_id}s and no updates. Loading existing feature parquet and stopping: {save_path}")
                        return fe_cached, _build_feature_groups(fe_cached, targets or list(DEFAULT_TARGETS))
                else:
                    print(f"Detected {len(new_ids)} new {in_id}(s) → recomputing features.")
            else:
                print(f"'{id_col}' not present in both inputs (checked upper/lower); proceeding to recompute features.")
    
    print("Running full feature engineering pipeline")
    fe = gamelogs.copy()

    # ---------------- Required columns: create safe defaults if missing ----------------
    # Dates
    if "game_date" not in fe.columns:
        raise KeyError("Missing required column 'game_date' in gamelogs.")
    fe["game_date"] = pd.to_datetime(fe["game_date"], errors="coerce")

    # Ensure team_winner exists (default 0 if missing)
    if "team_winner" not in fe.columns:
        fe["team_winner"] = 0

    # Ensure starter_injured exists and is boolean/numeric
    if "starter_injured" not in fe.columns:
        fe["starter_injured"] = False
    fe["starter_injured"] = fe["starter_injured"].fillna(False).astype(np.int32)

    # Ensure starter exists
    if "starter" not in fe.columns:
        fe["starter"] = False

    # Optional categorical columns that are used later
    if "home_away" not in fe.columns:
        # create a neutral category to avoid dropped rows in home/away features
        fe["home_away"] = "unknown"

    if "season_type" not in fe.columns:
        # downstream expects an int-coded season_type; default to Regular Season-like (0)
        fe["season_type"] = 0

    # ---------------- High-level filters & opponent aggregates ----------------
    # Drop All-Star Games
    all_star_teams = ["DUR", "LEB", "GIA", "WEST", "EAST", "CAN", "CHK", "KEN", "SHQ"]
    if "team_abbreviation" in fe.columns:
        fe = fe[~fe["team_abbreviation"].isin(all_star_teams)].reset_index(drop=True)

    # Sort for cumulative/rolling logic
    sort_keys = ["opponent_team_abbreviation", "season", "game_date"]
    sort_keys = [c for c in sort_keys if c in fe.columns]
    if len(sort_keys) == 3:
        fe = fe.sort_values(sort_keys).reset_index(drop=True)

    # Opponent cumulative losses/wins and win%
    if {"opponent_team_abbreviation", "season"}.issubset(fe.columns):
        fe["opp_cum_losses"] = (
            (fe["team_winner"] == 1)
            .groupby([fe["opponent_team_abbreviation"], fe["season"]])
            .cumsum()
            .astype(np.int32)
        )
        fe["opp_cum_wins"] = (
            (fe["team_winner"] == 0)
            .groupby([fe["opponent_team_abbreviation"], fe["season"]])
            .cumsum()
            .astype(np.int32)
        )
        denom = (fe["opp_cum_wins"] + fe["opp_cum_losses"]).replace(0, np.nan)
        fe["opp_win_pct"] = fe["opp_cum_wins"] / denom

        # Opp days since last + b2b
        fe["opp_days_since_last_game"] = (
            fe.groupby(["opponent_team_abbreviation", "season"])["game_date"]
            .diff()
            .dt.days.astype("float32")
        )
        fe["opp_is_back_to_back"] = (fe["opp_days_since_last_game"] == 1).astype(np.int32)

        # Opp games in last 7d
        opponent_games = (
            fe[["opponent_team_abbreviation", "season", "game_date"]]
            .drop_duplicates()
            .sort_values(["opponent_team_abbreviation", "season", "game_date"])
            .copy()
        )
        opponent_games["opp_games_last_7d"] = (
            opponent_games.groupby(["opponent_team_abbreviation", "season"], group_keys=False)
            .apply(lambda g: pd.Series(
                g.set_index("game_date").rolling("7D")["opponent_team_abbreviation"].count().values,
                index=g.index
            ))
            .astype("float32")
        )
        fe = fe.merge(
            opponent_games,
            on=["opponent_team_abbreviation", "season", "game_date"],
            how="left",
        )

    # Global temporal sort for all player-based rollups
    base_sort = [c for c in ["athlete_display_name", "game_date", "season", "team_abbreviation"] if c in fe.columns]
    if len(base_sort) >= 2:
        fe = fe.sort_values(base_sort).reset_index(drop=True)
        
    # Num starters injured that day (safe if team/date exist)
    if {"game_date", "team_abbreviation"}.issubset(fe.columns):
        fe["num_starters_injured"] = (
            fe.groupby(["game_date", "team_abbreviation"])["starter_injured"].transform("sum")
        ).astype("float32")
    else:
        fe["num_starters_injured"] = 0.0

    # plus_minus cleanup (if present)
    if "plus_minus" in fe.columns:
        fe["plus_minus"] = (
            fe["plus_minus"]
            .astype(str)
            .str.replace(r"^\+", "", regex=True)
            .replace("None", np.nan)
            .astype("float32")
        )

    # Ensure numeric dtypes / booleans
    fe["starter"] = fe["starter"].fillna(False).astype(int)
    for col in ["starter", "ejected", "team_winner", "is_playoff", "starter_injured"]:
        if col in fe.columns:
            fe[col] = fe[col].fillna(0).astype(np.int32)

    # Playoff indicator (your mapping if season_type provided)
    fe["is_playoff"] = fe.get("season_type", pd.Series(0, index=fe.index)).isin([3, 5]).astype(np.int32)

    # Days since last game + player b2b
    if {"athlete_display_name", "season"}.issubset(fe.columns):
        fe["days_since_last_game"] = (
            fe.groupby(["athlete_display_name", "season"])["game_date"].diff().dt.days.astype("float32")
        )
    else:
        fe["days_since_last_game"] = np.float32(0)
    fe["is_back_to_back"] = (fe["days_since_last_game"].fillna(99) == 1).astype(np.int32)

    # Shooting % (safe divides)
    def _safe_ratio(num, den):
        return (fe[num] / fe[den].replace(0, np.nan)).astype(float) if {num, den}.issubset(fe.columns) else np.nan

    fe["fg_pct"]  = _safe_ratio("field_goals_made", "field_goals_attempted")
    fe["fg3_pct"] = _safe_ratio("three_point_field_goals_made", "three_point_field_goals_attempted")
    fe["ft_pct"]  = _safe_ratio("free_throws_made", "free_throws_attempted")

    # Rolling/EWM base columns
    rolling_cols = [
        # Core performance
        "minutes",
        "field_goals_made",
        "field_goals_attempted",
        "three_point_field_goals_made",
        "three_point_field_goals_attempted",
        "free_throws_made",
        "free_throws_attempted",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "points",
        # Advanced team context
        "off_rating",
        "def_rating",
        "ast_pct",
        "oreb_pct",
        "dreb_pct",
        "reb_pct",
        "efg_pct",
        "ts_pct",
        "usg_pct",
        "pace",
        "poss",
        # On/Off
        "on_points",
        "on_fga",
        "on_fgm",
        "on_fg3a",
        "on_fg3m",
        "on_offensive_rebounds",
        "on_defensive_rebounds",
        "on_assists",
        "on_turnovers",
        "off_points",
        "off_fga",
        "off_fgm",
        "off_fg3a",
        "off_fg3m",
        "off_offensive_rebounds",
        "off_defensive_rebounds",
        "off_assists",
        "off_turnovers",
        # Lineups
        "total_lineup_minutes",
        "num_unique_lineups",
        "avg_lineup_ppm",
    ]
    # Keep only those that exist to avoid KeyErrors
    rolling_cols = [c for c in rolling_cols if c in fe.columns]
    lag_cols = rolling_cols + ([c for c in ["ejected"] if c in fe.columns])

    # --- Lag features ---
    def _lag_one(df: pd.DataFrame, col: str) -> pd.DataFrame:
        d = df.sort_values(["athlete_display_name", "game_date"])
        out = pd.DataFrame(index=d.index)
        out[f"{col}_lag1"] = d.groupby(["athlete_display_name"])[col].shift(1)
        out[f"{col}_lag2"] = d.groupby(["athlete_display_name"])[col].shift(2)
        return out

    if rolling_cols:
        lag_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_lag_one)(fe, col) for col in lag_cols
        )
        if lag_results:
            fe = pd.concat([fe] + lag_results, axis=1)

    # --- Expanding features ---
    def _expanding(df: pd.DataFrame, col: str) -> pd.DataFrame:
        d = df.sort_values(["athlete_display_name", "season", "game_date"])
        out = pd.DataFrame(index=d.index)
        shifted = d.groupby(["athlete_display_name"])[col].shift(1)
        g = shifted.groupby(d["athlete_display_name"])
        out[f"{col}_expanding_mean"] = g.expanding().mean().reset_index(level=[0, 1], drop=True)
        out[f"{col}_expanding_std"]  = g.expanding().std().reset_index(level=[0, 1], drop=True)
        out[f"{col}_expanding_max"]  = g.expanding().max().reset_index(level=[0, 1], drop=True)
        out[f"{col}_expanding_min"]  = g.expanding().min().reset_index(level=[0, 1], drop=True)
        tmp = d[["game_date"]].copy()
        tmp[f"{col}_expanding_mean"] = out[f"{col}_expanding_mean"]
        out[f"{col}_expanding_mean_rank"] = (
            tmp.groupby("game_date")[f"{col}_expanding_mean"].transform(lambda x: x.rank(pct=True)).astype("float32")
        )
        return out

    if rolling_cols:
        expanding_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_expanding)(fe, col) for col in rolling_cols
        )
        if expanding_results:
            fe = pd.concat([fe] + expanding_results, axis=1)

    # --- Rolling features ---
    def _rolling(df: pd.DataFrame, col: str, window: int = 5) -> pd.DataFrame:
        d = df.sort_values(["athlete_display_name", "game_date"]).copy()
        out = pd.DataFrame(index=d.index)
        shifted = d.groupby("athlete_display_name")[col].shift(1)
        g = shifted.groupby(d["athlete_display_name"])
        rmean = g.rolling(window).mean().reset_index(level=0, drop=True)
        out[f"{col}_rolling_mean"] = rmean
        out[f"{col}_rolling_std"]  = g.rolling(window).std().reset_index(level=0, drop=True)
        out[f"{col}_rolling_max"]  = g.rolling(window).max().reset_index(level=0, drop=True)
        out[f"{col}_rolling_min"]  = g.rolling(window).min().reset_index(level=0, drop=True)
        out[f"{col}_rolling_mad"]  = g.rolling(window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).reset_index(level=0, drop=True)
        out[f"{col}_rolling_trend"] = g.rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan, raw=False
        ).reset_index(level=0, drop=True)
        tmp = d[["game_date"]].copy()
        tmp[f"{col}_rolling_mean"] = rmean
        out[f"{col}_rolling_mean_rank"] = (
            tmp.groupby("game_date")[f"{col}_rolling_mean"].transform(lambda x: x.rank(pct=True).astype("float32"))
        )
        return out

    if rolling_cols:
        rolling_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_rolling)(fe, col, window=5) for col in rolling_cols
        )
        if rolling_results:
            fe = pd.concat([fe] + rolling_results, axis=1)

    # --- EWM features ---
    def _ewm(df: pd.DataFrame, col: str, spans=(21,)) -> pd.DataFrame:
        d = df.sort_values(["athlete_display_name", "game_date"])
        out = pd.DataFrame(index=d.index)
        for sp in spans:
            shifted = d.groupby("athlete_display_name")[col].shift(1)
            g = shifted.groupby(d["athlete_display_name"])
            ewm_mean = g.ewm(span=sp, adjust=False).mean().reset_index(level=0, drop=True)
            ewm_std  = g.ewm(span=sp, adjust=False).std().reset_index(level=0, drop=True)
            out[f"{col}_ewm_mean_span{sp}"] = ewm_mean
            out[f"{col}_ewm_std_span{sp}"]  = ewm_std
            out[f"{col}_ewm_zscore_span{sp}"] = (shifted - ewm_mean) / (ewm_std + 1e-6)
            out[f"{col}_ewm_above_avg_span{sp}"] = (shifted > ewm_mean).astype(int)
            surge = (shifted > ewm_mean).astype(int)
            out[f"{col}_ewm_surge_count_span{sp}"] = surge.groupby(d["athlete_display_name"]).cumsum()
            out[f"{col}_ewm_reversion_score_span{sp}"] = (shifted - ewm_mean).abs() / (ewm_std + 1e-6)
            out[f"{col}_ewm_mad_span{sp}"] = g.transform(
                lambda x: np.abs(x - x.ewm(span=sp, adjust=False).mean())
            ).reset_index(level=0, drop=True)
            tmp = d[["game_date"]].copy()
            tmp[f"{col}_ewm_mean_span{sp}"] = ewm_mean
            out[f"{col}_ewm_mean_span{sp}_rank"] = (
                tmp.groupby("game_date")[f"{col}_ewm_mean_span{sp}"].transform(lambda x: x.rank(pct=True)).astype("float32")
            )
        return out

    if rolling_cols:
        ewm_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_ewm)(fe, col) for col in rolling_cols
        )
        if ewm_results:
            fe = pd.concat([fe] + ewm_results, axis=1)

    # --- Delta features ---
    def _delta(df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        e_mean = f"{col}_expanding_mean"
        lag1 = f"{col}_lag1"
        if e_mean in df.columns and lag1 in df.columns:
            out[f"{col}_delta_lag1_vs_expanding"] = df[lag1] - df[e_mean]
        return out

    if rolling_cols:
        delta_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_delta)(fe, col) for col in rolling_cols
        )
        if delta_results:
            fe = pd.concat([fe] + delta_results, axis=1)

    # --- Home/Away EWM for shooting cols ---
    shooting_cols = [
        "points",
        "field_goals_made",
        "field_goals_attempted",
        "three_point_field_goals_made",
        "three_point_field_goals_attempted",
    ]
    shooting_cols = [c for c in shooting_cols if c in fe.columns]

    def _homeaway(df: pd.DataFrame, col: str, sp=21) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for ctx in ["home", "away"]:
            sub = df[df["home_away"] == ctx].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["athlete_display_name", "season", "game_date"])
            shifted = sub.groupby("athlete_display_name")[col].shift(1)
            ewm_mean = shifted.groupby(sub["athlete_display_name"]).ewm(span=sp, adjust=False).mean().reset_index(level=0, drop=True)
            ewm_col = f"{col}_ewm_mean_{ctx}_span{sp}"
            out.loc[sub.index, ewm_col] = ewm_mean
            rank_col = f"{ewm_col}_rank"
            out.loc[sub.index, rank_col] = (
                ewm_mean.groupby(sub["game_date"]).transform(lambda x: x.rank(pct=True).astype("float32"))
            )
        return out

    if shooting_cols:
        homeaway_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_homeaway)(fe, col) for col in shooting_cols
        )
        if homeaway_results:
            fe = pd.concat([fe] + homeaway_results, axis=1)

    # --- EWM trend slope per player/season ---
    def _trend_for_group(name, group, col, sp):
        g = group.sort_values("game_date")
        vals = g[col].shift(1).to_numpy()
        idx = g.index.to_numpy()
        slopes = np.full(len(vals), np.nan, dtype=np.float32)
        for i in range(sp - 1, len(vals)):
            y = vals[: i + 1]
            x = np.arange(len(y)).reshape(-1, 1)
            mask = ~np.isnan(y)
            if np.count_nonzero(mask) >= 3 and not np.allclose(y[mask], y[mask][0]):
                try:
                    weights = pd.Series(np.ones(len(y))).ewm(span=sp, adjust=False).mean().to_numpy()
                    weights = weights[mask]
                    model = LinearRegression()
                    model.fit(x[mask], y[mask], sample_weight=weights)
                    slopes[i] = model.coef_[0]
                except Exception:
                    pass
        return idx, slopes

    if rolling_cols and {"athlete_display_name", "season"}.issubset(fe.columns):
        for sp in [21]:
            for col in rolling_cols:
                groups = fe.groupby(["athlete_display_name", "season"])
                results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
                    delayed(_trend_for_group)(name, grp, col, sp) for name, grp in groups
                )
                slope_arr = np.full(len(fe), np.nan, dtype=np.float32)
                for idxs, slopes in results:
                    slope_arr[idxs] = slopes
                slope_col = f"{col}_ewm_trend_slope_{sp}"
                fe[slope_col] = slope_arr
                fe[f"{slope_col}_rank"] = (
                    fe.groupby("game_date")[slope_col].transform(lambda x: x.rank(pct=True).astype("float32"))
                )

    # Hot/cold flags + streaks
    if {"points", "points_ewm_mean_span21", "points_ewm_std_span21"}.issubset(fe.columns):
        fe["is_hot_game"] = (
            (fe["points"] >= fe["points_ewm_mean_span21"] + fe["points_ewm_std_span21"]).fillna(False).astype(int)
        )
        fe["is_cold_game"] = (
            (fe["points"] < fe["points_ewm_mean_span21"] - fe["points_ewm_std_span21"]).fillna(False).astype(int)
        )

        if {"athlete_display_name", "season"}.issubset(fe.columns):
            fe["hot_streak"] = (
                fe.sort_values(["athlete_display_name", "season", "game_date"])
                .groupby(["athlete_display_name", "season"])["is_hot_game"]
                .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
                .astype(np.int32)
            )
            fe["cold_streak"] = (
                fe.sort_values(["athlete_display_name", "season", "game_date"])
                .groupby(["athlete_display_name", "season"])["is_cold_game"]
                .transform(lambda x: x.shift(1).groupby((x.shift(1) != 1).cumsum()).cumcount())
                .astype(np.int32)
            )
        fe = fe.drop(columns=["is_hot_game", "is_cold_game"], errors="ignore")

    # Opponent EWM allowed (points/reb/3PM/to)
    opponent_stats = ["team_score", "rebounds", "three_point_field_goals_made", "turnovers"]
    ewm_spans = [21]

    def _opp_allowed(stat: str, sp: int) -> pd.DataFrame:
        if not {"opponent_team_abbreviation", "season", "game_date", stat}.issubset(fe.columns):
            return pd.DataFrame()
        daily = (
            fe.groupby(["opponent_team_abbreviation", "season", "game_date"])[stat]
            .sum()
            .reset_index()
            .sort_values(["opponent_team_abbreviation", "season", "game_date"])
        )
        ewm_col = f"opponent_ewm_{stat}_allowed_span{sp}"
        daily[ewm_col] = (
            daily.groupby(["opponent_team_abbreviation", "season"])[stat]
            .transform(lambda x: x.shift(1).ewm(span=sp, adjust=False).mean())
        )
        rank_col = f"{ewm_col}_pct_rank"
        daily[rank_col] = daily.groupby("game_date")[ewm_col].transform(lambda x: x.rank(pct=True).astype("float32"))
        return daily[["opponent_team_abbreviation", "season", "game_date", ewm_col, rank_col]]

    if {"opponent_team_abbreviation", "season", "game_date"}.issubset(fe.columns):
        opp_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_opp_allowed)(stat, sp) for stat in opponent_stats for sp in ewm_spans
        )
        for res in opp_results:
            if not res.empty:
                fe = fe.merge(res, on=["opponent_team_abbreviation", "season", "game_date"], how="left")

        fe = fe.rename(
            columns={
                "opponent_ewm_team_score_allowed_span21_pct_rank": "opponent_ewm_points_allowed_span21_pct_rank",
                "opponent_ewm_team_score_allowed_span21": "opponent_ewm_points_allowed_span21",
            }
        )

    # Opponent FG% / 3P% allowed (EWM)
    def _opp_pct(sp: int) -> pd.DataFrame:
        needed = [
            "field_goals_made",
            "field_goals_attempted",
            "three_point_field_goals_made",
            "three_point_field_goals_attempted",
        ]
        base_keys = {"opponent_team_abbreviation", "season", "game_date"}
        if not (set(needed).issubset(fe.columns) and base_keys.issubset(fe.columns)):
            return pd.DataFrame()

        daily = (
            fe.groupby(["opponent_team_abbreviation", "season", "game_date"])[needed]
            .sum()
            .reset_index()
            .sort_values(["opponent_team_abbreviation", "season", "game_date"])
        )
        for made_col, att_col in [
            ("field_goals_made", "field_goals_attempted"),
            ("three_point_field_goals_made", "three_point_field_goals_attempted"),
        ]:
            daily[f"{made_col}_shift"] = daily.groupby(["opponent_team_abbreviation", "season"])[made_col].shift(1)
            daily[f"{att_col}_shift"] = daily.groupby(["opponent_team_abbreviation", "season"])[att_col].shift(1)

        daily["ewm_fgm"] = (
            daily.groupby(["opponent_team_abbreviation", "season"])["field_goals_made_shift"]
            .transform(lambda x: x.ewm(span=sp, adjust=False).mean())
        )
        daily["ewm_fga"] = (
            daily.groupby(["opponent_team_abbreviation", "season"])["field_goals_attempted_shift"]
            .transform(lambda x: x.ewm(span=sp, adjust=False).mean())
        )
        daily["ewm_3pm"] = (
            daily.groupby(["opponent_team_abbreviation", "season"])["three_point_field_goals_made_shift"]
            .transform(lambda x: x.ewm(span=sp, adjust=False).mean())
        )
        daily["ewm_3pa"] = (
            daily.groupby(["opponent_team_abbreviation", "season"])["three_point_field_goals_attempted_shift"]
            .transform(lambda x: x.ewm(span=sp, adjust=False).mean())
        )

        daily["fg_pct"]  = daily["ewm_fgm"] / daily["ewm_fga"]
        daily["fg3_pct"] = daily["ewm_3pm"] / daily["ewm_3pa"]

        out = pd.DataFrame(
            {
                "opponent_team_abbreviation": daily["opponent_team_abbreviation"],
                "season": daily["season"],
                "game_date": daily["game_date"],
                f"opponent_ewm_fg_pct_allowed_span{sp}": daily["fg_pct"],
                f"opponent_ewm_fg3_pct_allowed_span{sp}": daily["fg3_pct"],
                f"opponent_ewm_fg_pct_allowed_rank_span{sp}": daily.groupby("game_date")["fg_pct"].transform(
                    lambda x: x.rank(pct=True).astype("float32")
                ),
                f"opponent_ewm_fg3_pct_allowed_rank_span{sp}": daily.groupby("game_date")["fg3_pct"].transform(
                    lambda x: x.rank(pct=True).astype("float32")
                ),
            }
        )
        return out

    if {"opponent_team_abbreviation", "season", "game_date"}.issubset(fe.columns):
        pct_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_opp_pct)(sp) for sp in ewm_spans
        )
        for res in pct_results:
            if not res.empty:
                fe = fe.merge(res, on=["opponent_team_abbreviation", "season", "game_date"], how="left")

    # Player x Opponent interactions
    def _interactions(df: pd.DataFrame, stat: str, sp: int) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        player_feat   = f"{stat}_ewm_mean_span{sp}"
        opp_rank_feat = f"opponent_ewm_{stat}_allowed_span{sp}_pct_rank"
        opp_raw_feat  = f"opponent_ewm_{stat}_allowed_span{sp}"
        if player_feat in df.columns and opp_rank_feat in df.columns:
            out[f"{player_feat}_x_opp_allowed_rank"] = df[player_feat] * df[opp_rank_feat]
        if player_feat in df.columns and opp_raw_feat in df.columns:
            out[f"{player_feat}_x_opp_allowed_raw"]  = df[player_feat] * df[opp_raw_feat]
        return out

    if {"opponent_team_abbreviation", "season", "game_date"}.issubset(fe.columns):
        interaction_stats = ["points", "rebounds", "turnovers", "three_point_field_goals_made"]
        inter_results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_interactions)(fe, s, 21) for s in interaction_stats
        )
        if inter_results:
            fe = pd.concat([fe] + inter_results, axis=1)

    # Team primary scorer flag
    if {"team_abbreviation", "game_date", "points_ewm_mean_span21"}.issubset(fe.columns):
        fe["team_primary_scorer"] = (
            fe.groupby(["team_abbreviation", "game_date"])["points_ewm_mean_span21"]
            .transform(lambda x: x == x.max())
            .astype(int)
        )

    # Drop temporary W/L cumulative columns
    fe = fe.drop(columns=["opp_cum_wins", "opp_cum_losses", "opp_win_pct"], errors="ignore")

    # Rolling team injuries
    if {"team_abbreviation", "season", "game_date"}.issubset(fe.columns):
        fe = (
            fe.sort_values(["team_abbreviation", "athlete_display_name", "season", "game_date"])
            .reset_index(drop=True)
        )
        fe["team_starter_injuries_rolling5"] = (
            fe.sort_values(["team_abbreviation", "season", "game_date"])
            .groupby(["team_abbreviation", "season"])["starter_injured"]
            .transform(lambda x: x.shift(1).rolling(5).sum())
            .astype("float32")
        )
    else:
        fe["team_starter_injuries_rolling5"] = np.float32(0)

    # Manual interactions
    def _manual(df: pd.DataFrame) -> pd.DataFrame:
        sp = 21
        pairs = [
            (f"minutes_ewm_mean_span{sp}", f"usg_pct_ewm_mean_span{sp}"),
            (f"points_ewm_mean_span{sp}", f"pace_ewm_mean_span{sp}"),
            (f"fg_pct_ewm_mean_span{sp}", f"usg_pct_ewm_mean_span{sp}"),
            (f"rebounds_ewm_mean_span{sp}", f"reb_pct_ewm_mean_span{sp}"),
            (f"assists_ewm_mean_span{sp}", f"ast_pct_ewm_mean_span{sp}"),
            (f"steals_ewm_mean_span{sp}", f"def_rating_ewm_mean_span{sp}"),
            (f"blocks_ewm_mean_span{sp}", f"def_rating_ewm_mean_span{sp}"),
            (f"rebounds_ewm_mean_span{sp}", f"reb_pct_ewm_mean_span{sp}"),
            (f"points_ewm_mean_span{sp}", f"opponent_ewm_points_allowed_span{sp}"),
            (f"points_ewm_zscore_span{sp}", f"pace_ewm_mean_span{sp}"),
            (f"points_ewm_mad_span{sp}", f"pace_ewm_mean_span{sp}"),
            (f"steals_ewm_zscore_span{sp}", f"def_rating_ewm_mean_span{sp}"),
            (f"three_point_field_goals_made_ewm_zscore_span{sp}", f"opponent_ewm_three_point_field_goals_made_allowed_span{sp}"),
            (f"rebounds_ewm_mad_span{sp}", f"reb_pct_ewm_mean_span{sp}"),
            (f"assists_ewm_zscore_span{sp}", f"ast_pct_ewm_mean_span{sp}"),
        ]
        out = pd.DataFrame(index=df.index)
        for v1, v2 in pairs:
            if v1 in df.columns and v2 in df.columns:
                out[f"{v1}_x_{v2}"] = df[v1] * df[v2]
        return out

    fe = pd.concat([fe, _manual(fe)], axis=1)

    # Downcast & clean
    float_cols = fe.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        fe[float_cols] = fe[float_cols].astype("float32")
    int_cols = fe.select_dtypes(include=["int64"]).columns
    if len(int_cols):
        fe[int_cols] = fe[int_cols].astype("int32")
    fe = fe.replace([np.inf, -np.inf], np.nan)

    # Save cache
    if cache_write:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fe.to_parquet(save_path, index=False)
        print(f"Saved feature-engineered gamelogs to {save_path}")

    # Build feature groups (same logic as your bottom block)
    feature_groups = _build_feature_groups(fe, targets)

    return fe, feature_groups


def _build_feature_groups(fe: pd.DataFrame, targets: List[str]) -> Dict[str, List[str]]:
    # Lag/rolling/expanding/etc. features
    lagged_rolling_features = [
        c
        for c in fe.columns
        if any(k in c for k in ["rolling", "expanding", "trend", "lag", "streak", "span", "ewm"])
    ]

    # Priors: specific expanding stats for selected targets
    prior_features = [
        c
        for c in fe.columns
        if any(c == f"{t}_expanding_{stat}" for stat in ["mean", "std"] for t in targets)
    ]

    # Numeric features (exclude priors from the big set)
    numeric_features = [
        c
        for c in lagged_rolling_features
        + [
            "days_since_last_game",
            "num_starters_injured",
            "opp_games_last_7d",
            "opp_days_since_last_game"
        ]
        if c in fe.columns and c not in prior_features
    ]

    # Categorical & embeddings (common, exist in upstream)
    categorical_features = [
        "home_away",
        "athlete_position_abbreviation",
        "is_playoff",
        "starter_injured",
        "is_back_to_back",
        "opp_is_back_to_back",
        "team_primary_scorer"
    ]
    categorical_features = [c for c in categorical_features if c in fe.columns]

    embedding_features = ["athlete_display_name", "team_abbreviation", "opponent_team_abbreviation"]
    embedding_features = [c for c in embedding_features if c in fe.columns]

    # Final predictors (excluding embeddings & priors)
    features = list(dict.fromkeys(numeric_features + categorical_features))

    # Add missing-flag
    def add_generic_missing_flag(X_df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        X_df = X_df.copy()
        exist = [c for c in feature_list if c in X_df.columns]
        if not exist:
            X_df["was_missing"] = 0
        else:
            X_df["was_missing"] = X_df[exist].isna().any(axis=1).astype(int)
        return X_df

    fe2 = add_generic_missing_flag(fe, features)
    if "was_missing" not in fe.columns:
        fe["was_missing"] = fe2["was_missing"]

    categorical_features = list(dict.fromkeys(categorical_features + ["was_missing"]))
    features = list(dict.fromkeys(features + ["was_missing"]))

    return {
        "lagged_rolling_features": list(dict.fromkeys(lagged_rolling_features)),
        "prior_features": list(dict.fromkeys(prior_features)),
        "numeric_features": list(dict.fromkeys(numeric_features)),
        "categorical_features": list(dict.fromkeys(categorical_features)),
        "embedding_features": list(dict.fromkeys(embedding_features)),
        "features": list(dict.fromkeys(features)),
        "targets": targets,
        "targets2": list(DEFAULT_TARGETS2),  # helpful downstream
    }

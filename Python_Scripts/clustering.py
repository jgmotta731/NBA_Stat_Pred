# -*- coding: utf-8 -*-
# clustering.py
"""
Created on Sat Sep 27 2025
@author: jgmot
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

__all__ = ["ClusteringArtifacts", "run_clustering", "attach_clusters"]

@dataclass
class ClusteringArtifacts:
    pca_pipeline: Pipeline
    kmeans: KMeans
    stat_cols: List[str]
    train_player_summary: pd.DataFrame
    cluster_means: pd.DataFrame

def _build_stat_cols(gamelogs: pd.DataFrame) -> List[str]:
    targets_with_both = ["field_goals_made"]
    targets_mean_only = ["field_goals_attempted", "reb_pct", "ast_pct", "usg_pct", "poss",
                         "three_point_field_goals_attempted"]
    targets_std_only  = ["points", "assists", "rebounds", "steals", "blocks",
                         "three_point_field_goals_made"]

    stat_cols: List[str] = []
    for tgt in targets_with_both + targets_mean_only:
        cm = f"{tgt}_expanding_mean"
        if cm in gamelogs.columns: stat_cols.append(cm)
    for tgt in targets_with_both + targets_std_only:
        cs = f"{tgt}_expanding_std"
        if cs in gamelogs.columns: stat_cols.append(cs)
    return stat_cols

def attach_clusters(
    gamelogs: pd.DataFrame,
    pca_pipeline: Pipeline,
    kmeans: KMeans,
    stat_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply existing PCA+KMeans to assign cluster labels to all players."""
    if stat_cols is None:
        stat_cols = _build_stat_cols(gamelogs)
    full_player_summary = (
        gamelogs.sort_values(["athlete_display_name", "game_date"])
                .groupby("athlete_display_name")
                .tail(1)
                .set_index("athlete_display_name")[stat_cols]
                .reset_index()
    )
    X_full = pca_pipeline.transform(full_player_summary[stat_cols])
    full_player_summary["cluster_label"] = kmeans.predict(X_full) + 1

    player_clusters = full_player_summary[["athlete_display_name", "cluster_label"]].copy()
    player_clusters["cluster_label"] = player_clusters["cluster_label"].astype("float32")

    out = (
        gamelogs.drop(columns=["cluster_label"], errors="ignore")
                .merge(player_clusters, on="athlete_display_name", how="left")
                .copy()
    )
    # keep dtypes sensible (optional, safe)
    int64_cols = out.select_dtypes(include="int64").columns
    out[int64_cols] = out[int64_cols].astype("float32")
    return out

def run_clustering(
    gamelogs: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    *,
    seed: int = 42,
    train_season_cutoff: int = 2025,
    n_components: int = 2,
    k_final: int = 5,
    show_diagnostics: bool = False,
    persist: bool = True,
    datasets_dir: str = "datasets",
    models_dir: str = "models/clustering",
    pipelines_dir: str = "pipelines",
) -> Tuple[pd.DataFrame, Dict[str, List[str]], ClusteringArtifacts]:
    """
    Fit PCA+KMeans on players' latest expanding stats (train split) and attach cluster labels.

    Returns:
      (gamelogs_with_clusters, updated_feature_groups, artifacts)

    If persist=True (default), saves:
      - {datasets_dir}/gamelogs_ready_for_modeling.parquet
      - {models_dir}/nba_player_clustering.joblib
      - {pipelines_dir}/pca_pipeline.joblib
    """
    # -------------------- Build stat columns --------------------
    all_stat_cols = _build_stat_cols(gamelogs)
    train_gamelogs = gamelogs[gamelogs["season"] < train_season_cutoff].copy()
    stat_cols = [c for c in all_stat_cols if c in train_gamelogs.columns]
    if not stat_cols:
        raise ValueError("No stat columns for clustering. Check feature engineering outputs.")

    train_player_summary = (
        train_gamelogs.sort_values(["athlete_display_name", "game_date"])
                      .groupby("athlete_display_name")
                      .tail(1)
                      .set_index("athlete_display_name")[stat_cols]
                      .reset_index()
    )

    # -------------------- Preprocessing + PCA --------------------
    preprocessor = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())]), stat_cols)]
    )
    X_proc = preprocessor.fit_transform(train_player_summary[stat_cols])

    # optional diagnostics
    if show_diagnostics:
        import matplotlib.pyplot as plt
        import numpy as np
        pca_tmp = PCA(random_state=seed)
        _ = pca_tmp.fit_transform(X_proc)
        plt.figure(figsize=(8,4)); plt.plot(np.cumsum(pca_tmp.explained_variance_ratio_), marker="o")
        plt.title("PCA cumulative explained variance"); plt.xlabel("# PCs"); plt.ylabel("CumVar"); plt.grid(True); plt.show()

    pca_pipeline = Pipeline([("preprocessor", preprocessor),
                             ("pca", PCA(n_components=n_components, random_state=seed))])
    X_cluster_train = pca_pipeline.fit_transform(train_player_summary[stat_cols])

    # -------------------- KMeans --------------------
    kmeans = KMeans(n_clusters=k_final, random_state=seed)
    kmeans.fit(X_cluster_train)

    # For reference reporting
    train_player_summary = train_player_summary.copy()
    train_player_summary["cluster_label"] = kmeans.labels_ + 1
    cluster_means = train_player_summary.groupby("cluster_label")[stat_cols].mean().round(2).sort_index()

    # -------------------- Apply to ALL players --------------------
    out_gamelogs = attach_clusters(gamelogs, pca_pipeline, kmeans, stat_cols)

    # -------------------- Persist --------------------
    if persist:
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(pipelines_dir, exist_ok=True)
        out_gamelogs.to_parquet(os.path.join(datasets_dir, "gamelogs_ready_for_modeling.parquet"))
        joblib.dump(kmeans, os.path.join(models_dir, "nba_player_clustering.joblib"))
        joblib.dump(pca_pipeline, os.path.join(pipelines_dir, "pca_pipeline.joblib"))
        print("Saved clustering artifacts and gamelogs with clusters.")

    # -------------------- Update feature groups --------------------
    updated_groups = {k: list(v) for k, v in feature_groups.items()}
    updated_groups["features"] = list(dict.fromkeys(updated_groups.get("features", []) + ["cluster_label"]))
    updated_groups["categorical_features"] = list(
        dict.fromkeys(updated_groups.get("categorical_features", []) + ["cluster_label"])
    )

    artifacts = ClusteringArtifacts(
        pca_pipeline=pca_pipeline,
        kmeans=kmeans,
        stat_cols=stat_cols,
        train_player_summary=train_player_summary,
        cluster_means=cluster_means,
    )
    return out_gamelogs, updated_groups, artifacts

if __name__ == "__main__":
    # Minimal smoke test (expects engineered parquet at datasets/gamelogs_features.parquet)
    fe_path = "datasets/gamelogs_features.parquet"
    if os.path.exists(fe_path):
        gl = pd.read_parquet(fe_path)
        dummy_groups = {"features": [], "categorical_features": []}
        out, groups, arts = run_clustering(gl, dummy_groups, persist=False)
        print(out[["athlete_display_name","cluster_label"]].head())

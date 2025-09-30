# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 2025
@author: jgmot

Run end-to-end training:
1) scrape + load  -> gamelogs (gamelogs_ready_for_fe.parquet)
2) feature eng    -> gamelogs, feature_groups (gamelogs_features.parquet)
3) clustering     -> gamelogs (with cluster), feature_groups (gamelogs_ready_for_modeling.parquet)
4) preprocessing  -> tensors + artifacts
5) BNN training   -> results_df, artifacts
"""

from __future__ import annotations
import pandas as pd
from Python_Scripts.scraping_loading import run_scrape_and_load
from Python_Scripts.feature_engineering import run_feature_engineering
from Python_Scripts.clustering import run_clustering
from Python_Scripts.preprocessing import run_preprocessing
from Python_Scripts.bnn import run_bnn

pd.set_option('display.max_columns', 20)

def main() -> None:
    # 1) Scrape & load (writes/caches parquets internally)
    gamelogs = run_scrape_and_load()

    # 2) Feature engineering
    gamelogs, feature_groups = run_feature_engineering(gamelogs)

    # 3) PCA + KMeans clustering
    gamelogs, feature_groups, _ = run_clustering(
        gamelogs=gamelogs,
        feature_groups=feature_groups,
        seed=42,
        train_season_cutoff=2025,
        n_components=2,
        k_final=5,
        show_diagnostics=False,
    )

    # 4) Preprocessing (NOTE: arg name is season_cutoff, not train_season_cutoff)
    (
        X_train_proc,
        X_val_proc,
        y_train_scaled,
        y_val_scaled,
        prior_train_scaled,
        prior_val_scaled,
        X_train_embed,
        X_val_embed,
        pre_art,
    ) = run_preprocessing(
        gamelogs=gamelogs,
        feature_groups=feature_groups,
        season_cutoff=2025,
    )

    # 5) BNN training/eval (saves metrics internally if save_dir is set)
    results_df, _ = run_bnn(
        gamelogs=gamelogs,
        feature_groups=feature_groups,
        X_train_proc=X_train_proc,
        X_val_proc=X_val_proc,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        prior_train_scaled=prior_train_scaled,
        prior_val_scaled=prior_val_scaled,
        X_train_embed=X_train_embed,
        X_val_embed=X_val_embed,
        pre_art=pre_art,
        seed=42,
        batch_size=512,
        max_epochs=100,
        patience=3,
        aux_warmup=3,
        lambda_aux=0.2,
        lr=3e-4,
        mc_samples=50,
    )

    # Optional: echo results
    try:
        print("\n=== Validation metrics ===")
        print(results_df)
    except Exception:
        pass


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# preprocessing.py
"""
Created on Sat Sep 27 2025
@author: jgmot
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

__all__ = [
    "PreprocArtifacts",
    "run_preprocessing",
    "save_preproc_artifacts",
]

@dataclass
class PreprocArtifacts:
    # fitted transformers
    preprocessor: Pipeline
    prior_pipeline: Pipeline
    y_scaler: StandardScaler
    embed_encoder: OrdinalEncoder

    # embedding meta
    embedding_sizes: List[Tuple[int, int]]  # (num_embeddings, emb_dim)
    embedding_features: List[str]

    # columns / features actually used
    features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    prior_features: List[str]
    targets: List[str]
    targets2: List[str]

    # splits (optional, helpful for debugging/eval)
    train_index: pd.Index
    val_index: pd.Index


def _dedupe_keep_existing(cols: List[str], df: pd.DataFrame) -> List[str]:
    """De-duplicate while keeping only columns present in df."""
    return list(dict.fromkeys([c for c in cols if c in df.columns]))


def run_preprocessing(
    gamelogs: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    *,
    season_cutoff: int = 2025,   # train: season < cutoff, val: >= cutoff
    seed: int = 42,
    persist: bool = True,
    pipelines_dir: str = "pipelines",
) -> Tuple[
    np.ndarray, np.ndarray,           # X_train_proc, X_val_proc
    np.ndarray, np.ndarray,           # y_train_scaled, y_val_scaled
    np.ndarray, np.ndarray,           # prior_train_scaled, prior_val_scaled
    np.ndarray, np.ndarray,           # X_train_embed, X_val_embed
    PreprocArtifacts
]:
    """
    Builds train/val splits, encodes categoricals, OHEs + PCA numeric features,
    prepares priors, target scalers, and discrete embeddings.

    If persist=True (default), saves to {pipelines_dir}:
        - preprocessor_pipeline.joblib
        - prior_pipeline.joblib
        - embed_encoder.joblib
        - embedding_sizes.joblib
        - y_scaler.joblib

    Returns:
      X_train_proc, X_val_proc,
      y_train_scaled, y_val_scaled,
      prior_train_scaled, prior_val_scaled,
      X_train_embed, X_val_embed,
      artifacts (fitted transformers, sizes, and feature names)
    """

    # ----------- Pull groups (defensive to missing columns) -----------
    targets = feature_groups.get("targets", [
        "three_point_field_goals_made", "rebounds", "assists", "steals", "blocks", "points"
    ])
    targets2 = feature_groups.get("targets2", [
        "minutes", "field_goals_attempted", "field_goals_made",
        "free_throws_attempted", "free_throws_made",
        "three_point_field_goals_attempted",
        *targets,
    ])

    prior_features     = _dedupe_keep_existing(feature_groups.get("prior_features", []), gamelogs)
    numeric_features   = _dedupe_keep_existing(feature_groups.get("numeric_features", []), gamelogs)
    categorical_feats0 = _dedupe_keep_existing(feature_groups.get("categorical_features", []), gamelogs)
    embedding_features = _dedupe_keep_existing(
        feature_groups.get("embedding_features", ["athlete_display_name", "team_abbreviation", "opponent_team_abbreviation"]),
        gamelogs
    )

    # Final predictors (exclude priors & embeddings by design)
    features = _dedupe_keep_existing(feature_groups.get("features", numeric_features + categorical_feats0), gamelogs)

    # ----------- Temporal split -----------
    train_df = gamelogs[gamelogs["season"] < season_cutoff].copy()
    val_df   = gamelogs[gamelogs["season"] >= season_cutoff].copy()

    # Ensure categoricals are strings (stable OHE categories)
    categorical_features = list(categorical_feats0)
    for col in categorical_features:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].astype(str)

    # ----------- Embedding encoder (discrete) -----------
    # Ordinal -> shift +1 -> clamp; UNK/missing -> 0
    embed_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    if embedding_features:
        embed_encoder.fit(train_df[embedding_features])

        embedding_sizes: List[Tuple[int, int]] = []
        for cats in embed_encoder.categories_:
            n_seen = len(cats)
            emb_dim = min(50, max(4, int(np.sqrt(n_seen + 1))))
            embedding_sizes.append((n_seen + 1, emb_dim))

        X_train_embed = embed_encoder.transform(train_df[embedding_features]).astype(np.int64) + 1
        X_val_embed   = embed_encoder.transform(val_df[embedding_features]).astype(np.int64) + 1

        # clamp into [0, num_embeddings-1]
        for j, (num_embeddings, _) in enumerate(embedding_sizes):
            X_train_embed[:, j] = np.clip(X_train_embed[:, j], 0, num_embeddings - 1)
            X_val_embed[:, j]   = np.clip(X_val_embed[:, j],   0, num_embeddings - 1)

        # quick diagnostics (caller can log if desired)
        _ = {embedding_features[j]: float((X_val_embed[:, j] == 0).mean()) for j in range(len(embedding_features))}
    else:
        embedding_sizes = []
        X_train_embed = np.zeros((len(train_df), 0), dtype=np.int64)
        X_val_embed   = np.zeros((len(val_df), 0), dtype=np.int64)

    # ----------- Targets (scaled) -----------
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(train_df[targets])
    y_val_scaled   = y_scaler.transform(val_df[targets])

    # ----------- Priors (expanding means/stds) -----------
    if prior_features:
        prior_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ])
        prior_train_scaled = prior_pipeline.fit_transform(train_df[prior_features])
        prior_val_scaled   = prior_pipeline.transform(val_df[prior_features])
    else:
        prior_pipeline = Pipeline([("passthrough", "passthrough")])
        prior_train_scaled = np.zeros((len(train_df), 0), dtype=np.float32)
        prior_val_scaled   = np.zeros((len(val_df), 0), dtype=np.float32)

    # ----------- Predictors -> ColumnTransformer -----------
    numeric_features       = [c for c in numeric_features if c in features]
    categorical_features   = [c for c in categorical_features if c in features]

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])

    standard_numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=seed)),  # retain ~95% variance
    ])

    column_transformer = ColumnTransformer([
        ("num", standard_numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    preprocessor = Pipeline([
        ("transform", column_transformer),
        ("var_thres", VarianceThreshold(threshold=1e-4)),
    ])

    # ----------- Fit / Transform -----------
    X_train = train_df[features]
    X_val   = val_df[features]

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)

    artifacts = PreprocArtifacts(
        preprocessor=preprocessor,
        prior_pipeline=prior_pipeline,
        y_scaler=y_scaler,
        embed_encoder=embed_encoder,
        embedding_sizes=embedding_sizes,
        embedding_features=embedding_features,
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        prior_features=prior_features,
        targets=list(targets),
        targets2=list(targets2),
        train_index=train_df.index,
        val_index=val_df.index,
    )

    # ----------- Persist artifacts on each run (default) -----------
    if persist:
        os.makedirs(pipelines_dir, exist_ok=True)
        joblib.dump(artifacts.preprocessor,    f"{pipelines_dir}/preprocessor_pipeline.joblib")
        joblib.dump(artifacts.prior_pipeline,  f"{pipelines_dir}/prior_pipeline.joblib")
        joblib.dump(artifacts.embed_encoder,   f"{pipelines_dir}/embed_encoder.joblib")
        joblib.dump(artifacts.embedding_sizes, f"{pipelines_dir}/embedding_sizes.joblib")
        joblib.dump(artifacts.y_scaler,        f"{pipelines_dir}/y_scaler.joblib")
        print(f"Saved preprocessing artifacts to '{pipelines_dir}'.")

    return (
        X_train_proc,
        X_val_proc,
        y_train_scaled,
        y_val_scaled,
        prior_train_scaled,
        prior_val_scaled,
        X_train_embed,
        X_val_embed,
        artifacts,
    )


def save_preproc_artifacts(
    artifacts: PreprocArtifacts,
    *,
    pipelines_dir: str = "pipelines",
) -> None:
    os.makedirs(pipelines_dir, exist_ok=True)
    joblib.dump(artifacts.preprocessor,    f"{pipelines_dir}/preprocessor_pipeline.joblib")
    joblib.dump(artifacts.prior_pipeline,  f"{pipelines_dir}/prior_pipeline.joblib")
    joblib.dump(artifacts.embed_encoder,   f"{pipelines_dir}/embed_encoder.joblib")
    joblib.dump(artifacts.embedding_sizes, f"{pipelines_dir}/embedding_sizes.joblib")
    joblib.dump(artifacts.y_scaler,        f"{pipelines_dir}/y_scaler.joblib")

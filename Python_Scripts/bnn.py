# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 2025
@author: jgmot

Reusable BNN module:
- QuantileBNN: model
- run_bnn: training + eval on temporal split (uses preproc artifacts)
- predict_mc: MC-dropout inference
- load_bnn_for_inference: tiny helper to restore weights for prediction
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_pinball_loss,
)

__all__ = [
    "BNNArtifacts",
    "QuantileBNN",
    "predict_mc",
    "run_bnn",
    "load_bnn_for_inference",
]

# ----------------------------- Public artifacts container -----------------------------
@dataclass
class BNNArtifacts:
    model: nn.Module
    best_state: Dict[str, torch.Tensor]
    y_scaler: object
    targets: List[str]
    train_index: pd.Index
    val_index: pd.Index
    embedding_sizes: List[Tuple[int, int]]

# ----------------------------- Helpers -----------------------------
def _to_numpy_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

# ----------------------------- Loss pieces -----------------------------
def soft_pinball_loss(preds, targets, tau=0.5, alpha=2.0):
    diff = targets - preds
    return torch.mean(F.softplus(alpha * (tau - (diff < 0).float()) * diff))

def kl_regularization(model, scale=1e-4):
    return sum((p ** 2).sum() for p in model.parameters()) * scale

def sharpness_penalty(lower, upper, target_width=5.0, scale=1e-3):
    w = (upper - lower).clamp(min=0.01)
    return scale * ((w - target_width) ** 2).mean()

def coverage_penalty(lower, upper, y_true, target=0.8, scale=1.0):
    covered = ((y_true >= lower) & (y_true <= upper)).float().mean()
    return scale * F.relu(target - covered)

def aux_breakout_bce(aux_probs: torch.Tensor, y_raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(y_raw) & torch.isfinite(tau)
    labels = (y_raw > tau).float()
    valid_f = valid.float()
    pos = (labels * valid_f).sum(0) / valid_f.sum(0).clamp_min(1.0)
    pos = pos.clamp(1e-6, 1 - 1e-6)
    w_pos, w_neg = (1.0 - pos), pos
    weights = (labels * w_pos + (1.0 - labels) * w_neg) * valid_f
    eps = 1e-6
    loss = - (weights * (labels * torch.log(aux_probs.clamp_min(eps)) +
                         (1 - labels) * torch.log((1 - aux_probs).clamp_min(eps)))).sum(0)
    denom = weights.sum(0).clamp_min(1.0)
    return (loss / denom).mean()

def total_loss(mean, lower, upper, median, logvar, y_true, model, weights,
               kl_scale=1e-4, sharp_scale=1e-3, coverage_scale=1.0):
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    var = torch.exp(logvar) + 1e-6
    hetero_loss = ((mean - y_true) ** 2 / var).mean() + var.mean()

    pinball_lower  = soft_pinball_loss(lower,  y_true, tau=0.1)
    pinball_upper  = soft_pinball_loss(upper,  y_true, tau=0.9)
    pinball_median = soft_pinball_loss(median, y_true, tau=0.5)

    kl    = kl_regularization(model, kl_scale)
    sharp = sharpness_penalty(lower, upper, scale=sharp_scale)
    cov   = coverage_penalty(lower, upper, y_true, scale=coverage_scale)

    return (weights * hetero_loss).mean() + pinball_lower + pinball_upper + pinball_median + kl + sharp + cov

# ----------------------------- Model -----------------------------
class QuantileBNN(nn.Module):
    def __init__(self, input_dim, embedding_sizes, prior_dim, output_dim=6, dropout_rate=0.3, aux_dim: int = 0):
        super().__init__()
        self.output_dim = output_dim
        self.aux_dim = aux_dim

        self.embedding_dropout = nn.Dropout(0.1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in embedding_sizes
        ])
        emb_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        self.input_base_dim = input_dim + emb_total
        self.norm_input = nn.LayerNorm(self.input_base_dim)

        self.shared_base = nn.Sequential(
            nn.Linear(self.input_base_dim, 256), nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ELU(), nn.Dropout(dropout_rate)
        )

        self.heads_mean   = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_lower  = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_upper  = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_median = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])
        self.heads_logvar = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_dim)])

        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(output_dim)])

        if self.aux_dim > 0:
            self.aux_logits = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.aux_dim)])

        # not used directly in loss; kept for parity with earlier versions
        self.register_buffer("temp_vector", torch.tensor([5.0, 5.5, 5.1, 9.1, 6.9, 4.5], dtype=torch.float32))

    def _encode(self, x_num, x_embed):
        emb = [self.embedding_dropout(e(x_embed[:, i])) for i, e in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1) if emb else x_num
        x = self.norm_input(x)
        return self.shared_base(x)

    def forward(self, x_num, x_embed, prior, use_prior=True):
        h = self._encode(x_num, x_embed)

        means, lowers, uppers, medians, logvars = [], [], [], [], []
        for i in range(self.output_dim):
            if use_prior and prior is not None and prior.shape[1] >= (i * 2 + 1):
                prior_mean = prior[:, i * 2 + 0].unsqueeze(1)
                a = torch.sigmoid(self.alpha[i])
                pred_mean = self.heads_mean[i](h)
                mean = a * pred_mean + (1 - a) * prior_mean
            else:
                mean = self.heads_mean[i](h)

            means.append(mean)
            lowers.append(self.heads_lower[i](h))
            uppers.append(self.heads_upper[i](h))
            medians.append(self.heads_median[i](h))
            logvars.append(self.heads_logvar[i](h))

        return (
            torch.cat(means, 1),
            torch.cat(lowers, 1),
            torch.cat(uppers, 1),
            torch.cat(medians, 1),
            torch.cat(logvars, 1),
        )

    def forward_aux(self, x_num, x_embed):
        if self.aux_dim == 0:
            raise RuntimeError("No aux heads configured.")
        h = self._encode(x_num, x_embed)
        probs = [torch.sigmoid(hd(h)) for hd in self.aux_logits]
        return torch.cat(probs, dim=1)

# ----------------------------- Inference helper -----------------------------
def predict_mc(model, X_num, X_embed, prior_tensor, T=20):
    device = next(model.parameters()).device
    X_num, X_embed, prior_tensor = X_num.to(device), X_embed.to(device), prior_tensor.to(device)

    model.train()  # enable dropout for MC
    means, lowers, uppers, medians = [], [], [], []

    for _ in range(T):
        with torch.no_grad():
            m, lo, up, med, _ = model(X_num, X_embed, prior_tensor, use_prior=True)
            means.append(m)
            lowers.append(lo)
            uppers.append(up)
            medians.append(med)

    means   = torch.stack(means)
    lowers  = torch.stack(lowers)
    uppers  = torch.stack(uppers)
    medians = torch.stack(medians)

    mean_mc   = means.mean(0)
    std_mc    = means.std(0)
    lower_q   = lowers.mean(0)
    upper_q   = uppers.mean(0)
    median_q  = medians.mean(0)

    return (
        mean_mc.cpu().numpy(),
        std_mc.cpu().numpy(),
        lower_q.cpu().numpy(),
        upper_q.cpu().numpy(),
        median_q.cpu().numpy(),
    )

def load_bnn_for_inference(
    weights_path: str,
    *,
    input_dim: int,
    embedding_sizes: List[Tuple[int, int]],
    prior_dim: int,
    output_dim: int,
    aux_dim: int = 0,
    device: Optional[str] = None,
) -> QuantileBNN:
    """
    Minimal helper to rebuild the architecture and load weights for inference.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileBNN(
        input_dim=input_dim,
        embedding_sizes=embedding_sizes,
        prior_dim=prior_dim,
        output_dim=output_dim,
        aux_dim=aux_dim,
    )
    state = torch.load(weights_path, map_location=device)
    # allow either full state dict or full serialized model (guard)
    if isinstance(state, dict) and all(k.startswith(("shared_base", "heads_", "embeddings", "alpha", "norm_input")) for k in state.keys()):
        model.load_state_dict(state)
    else:
        # fallback if someone saved the entire model; extract state_dict if possible
        try:
            model.load_state_dict(state.state_dict())  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"Unable to load weights from {weights_path}: {e}")
    model.to(device)
    model.eval()
    return model

# ----------------------------- Trainer -----------------------------
def run_bnn(
    gamelogs: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    # preproc outputs:
    X_train_proc, X_val_proc,
    y_train_scaled, y_val_scaled,
    prior_train_scaled, prior_val_scaled,
    X_train_embed, X_val_embed,
    pre_art,  # the PreprocArtifacts from preprocessing.py
    *,
    seed: int = 42,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 3,
    aux_warmup: int = 3,
    lambda_aux: float = 0.2,
    lr: float = 3e-4,
    mc_samples: int = 50,
) -> Tuple[pd.DataFrame, BNNArtifacts]:
    """
    Trains the QuantileBNN, evaluates on the temporal val split, and ALWAYS saves:
      - models/bnn/nba_bnn_full_model.pt
      - models/bnn/nba_bnn_weights_only.pt
      - datasets/Evaluation_Metrics.parquet
      - datasets/AuxBreakout_Probs.parquet
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets: List[str] = feature_groups.get("targets", [
        "three_point_field_goals_made", "rebounds", "assists", "steals", "blocks", "points"
    ])

    # ---- reconstruct train/val dataframes for tau/labels ----
    train_df = gamelogs.loc[pre_art.train_index]
    val_df   = gamelogs.loc[pre_art.val_index]

    break_targets = list(targets)
    tau_train = np.column_stack([
        train_df[f"{t}_expanding_mean"].to_numpy() + train_df[f"{t}_expanding_std"].to_numpy()
        for t in break_targets
    ]).astype("float32")
    tau_val = np.column_stack([
        val_df[f"{t}_expanding_mean"].to_numpy() + val_df[f"{t}_expanding_std"].to_numpy()
        for t in break_targets
    ]).astype("float32")

    y_break_train = train_df[break_targets].astype("float32").to_numpy()
    y_break_val   = val_df[break_targets].astype("float32").to_numpy()

    # ---- tensors ----
    X_train_np = _to_numpy_dense(X_train_proc)
    X_val_np   = _to_numpy_dense(X_val_proc)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val_np,   dtype=torch.float32)

    X_train_embed_tensor = torch.tensor(X_train_embed, dtype=torch.long)
    X_val_embed_tensor   = torch.tensor(X_val_embed,   dtype=torch.long)

    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val_scaled,   dtype=torch.float32)

    prior_train = torch.tensor(prior_train_scaled, dtype=torch.float32)
    prior_val   = torch.tensor(prior_val_scaled,   dtype=torch.float32)

    y_break_train_tensor = torch.tensor(y_break_train, dtype=torch.float32)
    y_break_val_tensor   = torch.tensor(y_break_val,   dtype=torch.float32)
    tau_train_tensor     = torch.tensor(tau_train,     dtype=torch.float32)
    tau_val_tensor       = torch.tensor(tau_val,       dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor, X_train_embed_tensor, y_train_tensor, prior_train,
            y_break_train_tensor, tau_train_tensor
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # ---- model ----
    prior_dim = prior_train.shape[1]
    model = QuantileBNN(
        input_dim=X_train_tensor.shape[1],
        embedding_sizes=pre_art.embedding_sizes,
        prior_dim=prior_dim,
        output_dim=y_train_tensor.shape[1],
        aux_dim=len(break_targets),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    target_weights = torch.tensor(
        [1.5 if t in ["points", "rebounds", "assists"] else 1.0 for t in targets],
        dtype=torch.float32,
        device=device,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state: Dict[str, torch.Tensor] = {}

    # ---- training loop ----
    for epoch in range(max_epochs):
        model.train()
        losses = []

        for xb_num, xb_emb, yb, priorb, yb_break, taub in train_loader:
            xb_num  = xb_num.to(device)
            xb_emb  = xb_emb.to(device)
            yb      = yb.to(device)
            priorb  = priorb.to(device)
            yb_break= yb_break.to(device)
            taub    = taub.to(device)

            use_prior = epoch >= aux_warmup

            mean, lower, upper, median, logvar = model(xb_num, xb_emb, priorb, use_prior=use_prior)
            loss_primary = total_loss(mean, lower, upper, median, logvar, yb, model, target_weights)

            if epoch >= aux_warmup:
                aux_probs = model.forward_aux(xb_num, xb_emb)
                loss_aux = aux_breakout_bce(aux_probs, yb_break, taub)
                loss = loss_primary + lambda_aux * loss_aux
            else:
                loss = loss_primary

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        # validation
        model.eval()
        with torch.no_grad():
            mean, lower, upper, median, logvar = model(
                X_val_tensor.to(device),
                X_val_embed_tensor.to(device),
                prior_val.to(device),
                use_prior=True,
            )
            val_loss = total_loss(mean, lower, upper, median, logvar, y_val_tensor.to(device), model, target_weights)

        print(f"Epoch {epoch+1} - Train Loss: {np.mean(losses):.4f} - Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ---- restore best and move to CPU ----
    model.load_state_dict(best_model_state)
    model.eval()
    model.to("cpu")

    # ---- stochastic forward (MC dropout) ----
    y_val_mean, y_val_std, y_val_lower, y_val_upper, y_val_median = predict_mc(
        model,
        X_val_tensor.cpu(),
        X_val_embed_tensor.cpu(),
        prior_val.cpu(),
        T=mc_samples,
    )

    # ---- inverse transform back to original target scale ----
    y_scaler = pre_art.y_scaler
    y_val_pred_unscaled   = y_scaler.inverse_transform(y_val_mean)
    y_val_std_unscaled    = y_val_std * y_scaler.scale_
    y_val_lower_unscaled  = y_scaler.inverse_transform(y_val_lower)
    y_val_upper_unscaled  = y_scaler.inverse_transform(y_val_upper)
    y_val_median_unscaled = y_scaler.inverse_transform(y_val_median)

    # ---- evaluation table ----
    results = []
    for i, target in enumerate(targets):
        y_true   = val_df[target].to_numpy()
        y_pred   = y_val_pred_unscaled[:, i]
        y_lower  = y_val_lower_unscaled[:, i]
        y_upper  = y_val_upper_unscaled[:, i]
        y_median = y_val_median_unscaled[:, i]

        rmse_mean = root_mean_squared_error(y_true, y_pred)
        mae_mean  = mean_absolute_error(y_true, y_pred)
        r2        = r2_score(y_true, y_pred)

        rmse_median = root_mean_squared_error(y_true, y_median)
        mae_median  = mean_absolute_error(y_true, y_median)

        pinball_10 = mean_pinball_loss(y_true, y_lower, alpha=0.1)
        pinball_50 = mean_pinball_loss(y_true, y_median, alpha=0.5)
        pinball_90 = mean_pinball_loss(y_true, y_upper, alpha=0.9)

        coverage = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))

        results.append({
            "Target": target,
            "RMSE_Mean": rmse_mean,
            "MAE_Mean": mae_mean,
            "R2": r2,
            "RMSE_Median": rmse_median,
            "MAE_Median": mae_median,
            "Pinball_10": pinball_10,
            "Pinball_50": pinball_50,
            "Pinball_90": pinball_90,
            "80pct_Coverage": coverage,
        })

    results_df = pd.DataFrame(results)

    # ---- ALWAYS SAVE to canonical paths ----
    import os
    os.makedirs("models/bnn", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    torch.save(model, "models/bnn/nba_bnn_full_model.pt")
    torch.save(model.state_dict(), "models/bnn/nba_bnn_weights_only.pt")
    results_df.to_parquet("datasets/Evaluation_Metrics.parquet", index=False)

    with torch.no_grad():
        aux_val_probs = model.forward_aux(
            X_val_tensor.cpu(), X_val_embed_tensor.cpu()
        ).numpy()
    pd.DataFrame(aux_val_probs, columns=targets).to_parquet(
        "datasets/AuxBreakout_Probs.parquet", index=False
    )

    artifacts = BNNArtifacts(
        model=model,
        best_state=best_model_state,
        y_scaler=y_scaler,
        targets=targets,
        train_index=pre_art.train_index,
        val_index=pre_art.val_index,
        embedding_sizes=pre_art.embedding_sizes,
    )
    return results_df, artifacts

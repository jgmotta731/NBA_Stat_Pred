# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 2025
@author: jgmot
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import os
import random
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
import torch
import torch.nn as nn
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
    calib_scales: np.ndarray  # per-target aleatoric scales learned post-hoc (STD)
    loss_name: str = "gaussian_nll_mc"


# ----------------------------- Helpers -----------------------------
def _to_numpy_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


# ----------------------------- Loss -----------------------------
def kl_regularization(model: nn.Module, scale: float = 1e-4) -> torch.Tensor:
    # Simple L2 over parameters as KL surrogate
    return sum((p ** 2).sum() for p in model.parameters()) * scale


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    y_true: torch.Tensor,
    target_weights: torch.Tensor,
    *,
    model: Optional[nn.Module] = None,
    kl_scale: float = 1e-4,
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL per target, then weighted.
    """
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    var = torch.exp(logvar) + 1e-6
    nll = 0.5 * (((y_true - mean) ** 2) / var + torch.log(var))  # [B, T]
    nll = (nll.mean(dim=0) * target_weights).mean()
    if (model is not None) and (kl_scale > 0):
        nll = nll + kl_regularization(model, kl_scale)
    return nll


# ----------------------------- Model -----------------------------
class QuantileBNN(nn.Module):
    """
    Mean/logvar heads; predictive quantiles come from MC sampling.
    """
    def __init__(self, input_dim, embedding_sizes, prior_dim, output_dim=6, dropout_rate=0.3, aux_dim: int = 0):
        super().__init__()
        self.output_dim = int(output_dim)
        self.aux_dim = int(aux_dim)

        # --- priors handling ---
        self.prior_dim = int(prior_dim)
        self.priors_per_target = max(1, self.prior_dim // max(1, self.output_dim)) if self.prior_dim > 0 else 0

        # --- embeddings ---
        self.embedding_dropout = nn.Dropout(0.1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in embedding_sizes
        ])
        emb_total = sum(emb_dim for _, emb_dim in embedding_sizes)
        self.input_base_dim = int(input_dim) + emb_total
        self.norm_input = nn.LayerNorm(self.input_base_dim)

        # --- shared trunk ---
        self.shared_base = nn.Sequential(
            nn.Linear(self.input_base_dim, 256), nn.LayerNorm(256), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ELU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ELU(), nn.Dropout(dropout_rate)
        )

        # --- target-specific heads (mean & log-variance only) ---
        self.heads_mean   = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.output_dim)])
        self.heads_logvar = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.output_dim)])

        # prior blend coefficient per target (sigmoid(alpha) ∈ (0,1))
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.output_dim)])

        # --- optional aux classification heads ---
        if self.aux_dim > 0:
            self.aux_logits = nn.ModuleList([nn.Linear(64, 1) for _ in range(self.aux_dim)])

    def _encode(self, x_num: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0:
            emb = [self.embedding_dropout(e(x_embed[:, i])) for i, e in enumerate(self.embeddings)]
            x = torch.cat([x_num] + emb, dim=1)
        else:
            x = x_num
        x = self.norm_input(x)
        return self.shared_base(x)

    def _select_prior_mean(self, prior: Optional[torch.Tensor], i: int) -> Optional[torch.Tensor]:
        if prior is None or self.priors_per_target == 0:
            return None
        P = prior.shape[1]
        if P == self.output_dim:
            return prior[:, i:i+1]
        base = i * self.priors_per_target
        if base < P:
            return prior[:, base:base+1]
        return None

    def forward(self, x_num, x_embed, prior=None, use_prior=True):
        h = self._encode(x_num, x_embed)
        means, logvars = [], []
        for i in range(self.output_dim):
            pred_mean = self.heads_mean[i](h)
            prior_mean = self._select_prior_mean(prior, i) if (use_prior and prior is not None) else None
            if use_prior and (prior_mean is not None):
                a = torch.sigmoid(self.alpha[i])
                mean = a * pred_mean + (1 - a) * prior_mean
            else:
                mean = pred_mean
            means.append(mean)
            logvars.append(self.heads_logvar[i](h))
        return torch.cat(means, 1), torch.cat(logvars, 1)

    def forward_aux(self, x_num, x_embed):
        if self.aux_dim == 0:
            raise RuntimeError("No aux heads configured.")
        h = self._encode(x_num, x_embed)
        probs = [torch.sigmoid(hd(h)) for hd in self.aux_logits]
        return torch.cat(probs, dim=1)


# ----------------------------- Inference helper -----------------------------
def predict_mc(
    model: nn.Module,
    X_num: torch.Tensor,
    X_embed: torch.Tensor,
    prior_tensor: torch.Tensor,
    T: int = 50,
    return_samples: bool = False,
    alea_scale: Optional[torch.Tensor] = None,
):
    """
    MC Dropout inference (mixture predictive distribution).
    Restores the model's original train/eval mode on exit.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.train()  # enable dropout for MC

    X_num, X_embed, prior_tensor = X_num.to(device), X_embed.to(device), prior_tensor.to(device)

    # Prepare aleatoric scaling (broadcast later)
    alea_scale_t = None
    if alea_scale is not None:
        alea_scale_t = alea_scale.to(device) if torch.is_tensor(alea_scale) else torch.tensor(alea_scale, dtype=torch.float32, device=device)

    means, logvars, samples = [], [], []
    try:
        with torch.no_grad():
            for _ in range(T):
                m, lv = model(X_num, X_embed, prior_tensor, use_prior=True)
                means.append(m)
                logvars.append(lv)
                eps = torch.randn_like(m)
                std = torch.exp(0.5 * lv)
                if alea_scale_t is not None:
                    std = std * alea_scale_t  # broadcast [1,TGT] over batch
                samples.append(m + std * eps)

        means   = torch.stack(means)        # [T,N,TGT]
        logvars = torch.stack(logvars)      # [T,N,TGT]
        samples = torch.stack(samples)      # [T,N,TGT]

        mean_pred     = means.mean(0)
        std_epistemic = means.std(0, unbiased=False)
        if alea_scale_t is None:
            std_aleatoric = torch.sqrt(torch.exp(logvars).mean(0))
        else:
            std_aleatoric = torch.sqrt((torch.exp(logvars) * (alea_scale_t ** 2)).mean(0))
        std_predictive = torch.sqrt(std_epistemic**2 + std_aleatoric**2)

        q10, q50, q90 = (torch.quantile(samples, q, dim=0) for q in (0.10, 0.50, 0.90))

        out = (
            mean_pred.cpu().numpy(),
            std_epistemic.cpu().numpy(),
            std_aleatoric.cpu().numpy(),
            std_predictive.cpu().numpy(),
            q10.cpu().numpy(), q50.cpu().numpy(), q90.cpu().numpy(),
        )
        if return_samples:
            return out + (samples.cpu().numpy(),)
        return out
    finally:
        model.train(was_training)


# ----------------------------- Calibration -----------------------------
def fit_std_scale_per_target(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    std_epi: np.ndarray,
    std_ale: np.ndarray,
    *,
    target_cov: float = 0.80,
    grid: Tuple[float, float, int] = (0.3, 3.0, 91),
    z: float = float(norm.ppf(0.9)),
) -> np.ndarray:
    """
    Fit per-target aleatoric scale s to hit target coverage for mean ± z * sqrt(Var_epi + (s*Var_ale)^2).
    """
    lo, hi, steps = grid
    ss = np.linspace(lo, hi, steps, dtype=np.float64)
    TGT = mean_pred.shape[1]
    scales = np.ones(TGT, dtype=np.float32)

    for i in range(TGT):
        mu = mean_pred[:, i]
        se = std_epi[:, i]
        sa = std_ale[:, i]
        best_s, best_err = 1.0, 1e9
        for s in ss:
            std_pred = np.sqrt(se**2 + (s * sa)**2)
            lower = mu - z * std_pred
            upper = mu + z * std_pred
            cov = float(np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper)))
            err = abs(cov - target_cov)
            if err < best_err:
                best_err, best_s = err, s
        scales[i] = np.float32(best_s)
    return scales


def fit_quantile_scale_per_target(
    y_true: np.ndarray,
    q50: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    *,
    target_cov: float = 0.80,
    grid: Tuple[float, float, int] = (0.3, 3.0, 91),
) -> np.ndarray:
    """
    Symmetric per-target scale s for MC quantiles around the median:
      q_lo' = q50 + s * (q_lo - q50),  q_hi' = q50 + s * (q_hi - q50)
    Fitted to hit target 80% coverage on the calibration subset.
    """
    lo, hi, steps = grid
    ss = np.linspace(lo, hi, steps, dtype=np.float64)
    TGT = q50.shape[1]
    scales = np.ones(TGT, dtype=np.float32)

    for i in range(TGT):
        med = q50[:, i]
        dlo = q_lo[:, i] - med
        dhi = q_hi[:, i] - med
        best_s, best_err = 1.0, 1e9
        for s in ss:
            lower = med + s * dlo
            upper = med + s * dhi
            cov = float(np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper)))
            err = abs(cov - target_cov)
            if err < best_err:
                best_err, best_s = err, s
        scales[i] = np.float32(best_s)
    return scales


# ----------------------------- Loader -----------------------------
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
    Rebuild architecture and load weights for inference.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileBNN(
        input_dim=input_dim,
        embedding_sizes=embedding_sizes,
        prior_dim=prior_dim,
        output_dim=output_dim,
        aux_dim=aux_dim,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        if hasattr(state, "state_dict"):
            model.load_state_dict(state.state_dict(), strict=False)
        else:
            raise

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
    pre_art,
    *,
    seed: int = 42,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 5,
    aux_warmup: int = 3,
    lambda_aux: float = 0.2,
    lr: float = 3e-4,
    train_mc_samples: int = 4,
    mc_samples: int = 100,
    val_mc_samples: int = 16,
    target_pi: float = 0.80,
) -> Tuple[pd.DataFrame, BNNArtifacts]:
    """
    Train, calibrate (STD then MC-quantile on held-out cal split), evaluate on eval split.
    Saves:
      - datasets/Evaluation_Metrics.parquet         (post-calibrated: std + quantile) [EVAL split]
      - datasets/Evaluation_Metrics_Uncal.parquet   (uncalibrated)                    [EVAL split]
      - datasets/Calibration_Scales.parquet         (per-target: Aleatoric_Scale, Quantile_Scale)
    """
    # ---- RNG & device ----
    os.environ.setdefault("PYTHONHASHSEED", str(seed))  # effective if set at process start
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- targets ----
    targets: List[str] = feature_groups.get("targets", [
        "three_point_field_goals_made", "rebounds", "assists", "steals", "blocks", "points"
    ])
    TGT = len(targets)

    # ---- reconstruct train/val dataframes for aux labels ----
    train_df = gamelogs.loc[pre_art.train_index]
    val_df   = gamelogs.loc[pre_art.val_index]

    break_targets = list(targets)
    tau_train = np.column_stack([
        train_df[f"{t}_expanding_mean"].to_numpy() + train_df[f"{t}_expanding_std"].to_numpy()
        for t in break_targets
    ]).astype("float32")
    y_break_train = train_df[break_targets].astype("float32").to_numpy()

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

    # --- sanity checks ---
    assert y_train_tensor.shape[1] == TGT, f"y_train has {y_train_tensor.shape[1]} targets, expected {TGT}."
    assert y_val_tensor.shape[1]   == TGT, f"y_val has {y_val_tensor.shape[1]} targets, expected {TGT}."

    y_break_train_tensor = torch.tensor(y_break_train, dtype=torch.float32)
    tau_train_tensor     = torch.tensor(tau_train,     dtype=torch.float32)

    # deterministic DataLoader RNG/worker seeds
    g = torch.Generator()
    g.manual_seed(seed)

    def _seed_worker(worker_id: int):
        ws = seed + worker_id
        np.random.seed(ws)
        random.seed(ws)
        torch.manual_seed(ws)

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor, X_train_embed_tensor, y_train_tensor, prior_train,
            y_break_train_tensor, tau_train_tensor
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=_seed_worker,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # ---- model ----
    prior_dim = int(prior_train.shape[1])
    model = QuantileBNN(
        input_dim=X_train_tensor.shape[1],
        embedding_sizes=pre_art.embedding_sizes,
        prior_dim=prior_dim,
        output_dim=y_train_tensor.shape[1],
        aux_dim=len(break_targets),
    ).to(device)

    assert model.output_dim == TGT, f"Model output_dim={model.output_dim} != number of targets={TGT}."

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- target weights: prioritize points > rebounds > others ----
    priority = {"points": 3.0, "rebounds": 2.0}
    base_w = 1.0
    target_weights = torch.tensor(
        [priority.get(t, base_w) for t in targets],
        dtype=torch.float32,
        device=device,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state: Dict[str, torch.Tensor] = {}

    # ---- training loop (MC-averaged per batch with grad accumulation) ----
    for epoch in range(max_epochs):
        model.train()
        train_loss_meter = []

        for xb_num, xb_emb, yb, priorb, yb_break, taub in train_loader:
            xb_num  = xb_num.to(device)
            xb_emb  = xb_emb.to(device)
            yb      = yb.to(device)
            priorb  = priorb.to(device)
            yb_break= yb_break.to(device)
            taub    = taub.to(device)

            use_prior = epoch >= aux_warmup

            optimizer.zero_grad(set_to_none=True)
            total_loss_accum = 0.0

            for _ in range(max(1, int(train_mc_samples))):
                mean, logvar = model(xb_num, xb_emb, priorb, use_prior=use_prior)  # dropout ON

                loss_primary = gaussian_nll_loss(
                    mean, logvar, yb, target_weights,
                    model=model, kl_scale=1e-4,
                )

                if epoch >= aux_warmup and model.aux_dim > 0:
                    eps = 1e-6
                    valid  = torch.isfinite(yb_break) & torch.isfinite(taub)
                    labels = (yb_break > taub).float()
                    valid_f = valid.float()

                    denom = valid_f.sum(0).clamp_min(1.0)
                    pos = (labels * valid_f).sum(0) / denom
                    pos = pos.clamp(1e-6, 1 - 1e-6)
                    w_pos, w_neg = (1.0 - pos), pos
                    weights = (labels * w_pos + (1.0 - labels) * w_neg) * valid_f

                    aux_probs = model.forward_aux(xb_num, xb_emb)
                    loss_aux = - (weights * (labels * torch.log(aux_probs.clamp_min(eps)) +
                                             (1 - labels) * torch.log((1 - aux_probs).clamp_min(eps)))).sum(0)
                    denom_w = weights.sum(0).clamp_min(1.0)
                    loss_aux = (loss_aux / denom_w).mean()

                    loss_k = loss_primary + lambda_aux * loss_aux
                else:
                    loss_k = loss_primary

                (loss_k / train_mc_samples).backward()
                total_loss_accum += float(loss_k.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_meter.append(total_loss_accum / max(1, int(train_mc_samples)))

        # ---- validation with MC-Dropout (matches inference) ----
        def _mc_val_loss(model, Xv, Xe, Pv, Yv, T=16):
            model.train()  # enable dropout
            ls = []
            with torch.no_grad():
                for _ in range(T):
                    m, lv = model(Xv.to(device), Xe.to(device), Pv.to(device), use_prior=True)
                    ls.append(gaussian_nll_loss(
                        m, lv, Yv.to(device), target_weights,
                        model=None, kl_scale=0.0
                    ).item())
            model.eval()
            return float(np.mean(ls))

        val_loss = _mc_val_loss(model, X_val_tensor, X_val_embed_tensor, prior_val, y_val_tensor, T=val_mc_samples)

        print(f"Epoch {epoch+1} - Train MC-NLL: {np.mean(train_loss_meter):.4f} - Val MC-NLL: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ---- restore best (keep on device for faster MC) ----
    model.load_state_dict(best_model_state)
    model.eval()

    # ====== Build deterministic CAL/EVAL split from validation ======
    N_val = y_val_tensor.shape[0]
    idx = np.arange(N_val)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(idx)
    cal_frac = 0.5 if N_val >= 4 else 0.5  # simple default
    split = max(1, int(cal_frac * N_val))
    cal_mask = np.zeros(N_val, dtype=bool); cal_mask[perm[:split]] = True
    eval_mask = ~cal_mask

    # ---- MC UNCALIBRATED on FULL val (device) ----
    (
        y_mean_unc, y_std_epi_unc, y_std_ale_unc, y_std_pred_unc,
        y_q10_unc, y_q50_unc, y_q90_unc
    ) = predict_mc(
        model,
        X_val_tensor, X_val_embed_tensor, prior_val,
        T=mc_samples,
    )

    # ---- STD calibration on CAL subset (standardized space) ----
    calib_scales = fit_std_scale_per_target(
        y_true=y_val_tensor.numpy()[cal_mask],
        mean_pred=y_mean_unc[cal_mask],
        std_epi=y_std_epi_unc[cal_mask],
        std_ale=y_std_ale_unc[cal_mask],
        target_cov=target_pi,
        grid=(0.3, 3.0, 91),
        z=float(norm.ppf(0.9)),
    )
    assert calib_scales.shape[0] == TGT, f"calib_scales length {calib_scales.shape[0]} != {TGT}."

    # ---- MC after STD calibration on FULL val (device) ----
    calib_scales_t = torch.tensor(calib_scales, dtype=torch.float32).view(1, -1)
    (
        y_mean_cal, y_std_epi_cal, y_std_ale_cal, y_std_pred_cal,
        y_q10_cal, y_q50_cal, y_q90_cal
    ) = predict_mc(
        model,
        X_val_tensor, X_val_embed_tensor, prior_val,
        T=mc_samples,
        alea_scale=calib_scales_t,
    )

    # ---- Quantile calibration on CAL subset (standardized space; symmetric around median) ----
    q_scales = fit_quantile_scale_per_target(
        y_true=y_val_tensor.numpy()[cal_mask],
        q50=y_q50_cal[cal_mask],
        q_lo=y_q10_cal[cal_mask],
        q_hi=y_q90_cal[cal_mask],
        target_cov=target_pi,
        grid=(0.3, 3.0, 91),
    )
    q_scales_t = q_scales.reshape(1, -1)

    # Apply quantile scaling to FULL val (still standardized space)
    y_q10_qcal = y_q50_cal + (y_q10_cal - y_q50_cal) * q_scales_t
    y_q90_qcal = y_q50_cal + (y_q90_cal - y_q50_cal) * q_scales_t
    y_q50_qcal = y_q50_cal  # keep median as-is

    # ---- inverse-transform helper (expects full arrays; we'll mask at eval time) ----
    y_scaler = pre_art.y_scaler
    if not hasattr(y_scaler, "scale_"):
        raise ValueError("y_scaler must expose a 'scale_' attribute (e.g., StandardScaler).")

    def _unscale_stats(mu, q10, q50, q90, std_e, std_a, std_p):
        mu_u   = y_scaler.inverse_transform(mu)
        q10_u  = y_scaler.inverse_transform(q10)
        q50_u  = y_scaler.inverse_transform(q50)
        q90_u  = y_scaler.inverse_transform(q90)
        std_eu = std_e * y_scaler.scale_
        std_au = std_a * y_scaler.scale_
        std_pu = std_p * y_scaler.scale_
        return mu_u, q10_u, q50_u, q90_u, std_eu, std_au, std_pu

    # ---- UNCALIBRATED (for EVAL subset only) ----
    (
        y_eval_pred_unc_u,
        y_eval_q10_unc_u,
        y_eval_q50_unc_u,
        y_eval_q90_unc_u,
        y_eval_std_epi_unc_u,
        y_eval_std_ale_unc_u,
        y_eval_std_pred_unc_u,
    ) = _unscale_stats(
        y_mean_unc[eval_mask], y_q10_unc[eval_mask], y_q50_unc[eval_mask], y_q90_unc[eval_mask],
        y_std_epi_unc[eval_mask], y_std_ale_unc[eval_mask], y_std_pred_unc[eval_mask]
    )

    # ---- POST-calibrated (STD + Quantile) (for EVAL subset only) ----
    (
        y_eval_pred_cal_u,
        _q10_std_u, _q50_std_u, _q90_std_u,
        y_eval_std_epi_cal_u,
        y_eval_std_ale_cal_u,
        y_eval_std_pred_cal_u,
    ) = _unscale_stats(
        y_mean_cal[eval_mask], y_q10_cal[eval_mask], y_q50_cal[eval_mask], y_q90_cal[eval_mask],
        y_std_epi_cal[eval_mask], y_std_ale_cal[eval_mask], y_std_pred_cal[eval_mask]
    )
    (
        _mu_tmp,
        y_eval_q10_qcal_u,
        y_eval_q50_qcal_u,
        y_eval_q90_qcal_u,
        _se_tmp, _sa_tmp, _sp_tmp
    ) = _unscale_stats(
        y_mean_cal[eval_mask], y_q10_qcal[eval_mask], y_q50_qcal[eval_mask], y_q90_qcal[eval_mask],
        y_std_epi_cal[eval_mask], y_std_ale_cal[eval_mask], y_std_pred_cal[eval_mask]
    )

    # ---- evaluation on EVAL subset ----
    eval_df = val_df.iloc[eval_mask.nonzero()[0]]

    def _evaluate(mu_u, q10_u, q50_u, q90_u, std_pred_u, std_epi_u, std_ale_u, label_suffix: str):
        results = []
        z80 = float(norm.ppf(0.9))
        for i, target in enumerate(targets):
            y_true   = eval_df[target].to_numpy()
            y_pred   = mu_u[:, i]
            y_median = q50_u[:, i]
            y_lower  = q10_u[:, i]
            y_upper  = q90_u[:, i]

            rmse_mean   = root_mean_squared_error(y_true, y_pred)
            mae_mean    = mean_absolute_error(y_true, y_pred)
            r2          = r2_score(y_true, y_pred)
            rmse_median = root_mean_squared_error(y_true, y_median)
            mae_median  = mean_absolute_error(y_true, y_median)

            pinball_10 = mean_pinball_loss(y_true, y_lower,  alpha=0.10)
            pinball_50 = mean_pinball_loss(y_true, y_median, alpha=0.50)
            pinball_90 = mean_pinball_loss(y_true, y_upper,  alpha=0.90)

            coverage_80    = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
            pi80_width     = float(np.mean(y_upper - y_lower))
            below_q10_rate = float(np.mean(y_true < y_lower))
            above_q90_rate = float(np.mean(y_true > y_upper))
            above_q50_rate = float(np.mean(y_true > y_median))

            std_pred = std_pred_u[:, i]
            lower_std80 = y_pred - z80 * std_pred
            upper_std80 = y_pred + z80 * std_pred
            coverage_std80 = float(np.mean((y_true >= lower_std80) & (y_true <= upper_std80)))

            mean_error = float(np.mean(y_pred - y_true))
            abs_err = np.abs(y_pred - y_true)
            if np.std(std_pred) > 0 and np.std(abs_err) > 0:
                uncert_error_corr = float(np.corrcoef(std_pred, abs_err)[0, 1])
            else:
                uncert_error_corr = np.nan

            std_pred_mean = float(np.mean(std_pred_u[:, i]))
            std_epi_mean  = float(np.mean(std_epi_u[:, i]))
            std_ale_mean  = float(np.mean(std_ale_u[:, i]))

            results.append({
                "Target": target,
                "Suffix": label_suffix,
                "RMSE_Mean": rmse_mean,
                "MAE_Mean": mae_mean,
                "R2": r2,
                "RMSE_Median": rmse_median,
                "MAE_Median": mae_median,
                "Pinball_10": pinball_10,
                "Pinball_50": pinball_50,
                "Pinball_90": pinball_90,
                "PI80_Coverage": coverage_80,
                "PI80_Width": pi80_width,
                "Below_Q10_Rate": below_q10_rate,
                "Above_Q90_Rate": above_q90_rate,
                "Above_Q50_Rate": above_q50_rate,
                "STD80_Coverage": coverage_std80,
                "Bias_MeanError": mean_error,
                "Uncert_Error_Corr": uncert_error_corr,
                "STD_Predictive_Mean": std_pred_mean,
                "STD_Epistemic_Mean": std_epi_mean,
                "STD_Aleatoric_Mean": std_ale_mean,
            })
        return pd.DataFrame(results)

    # Uncalibrated (EVAL)
    results_uncal = _evaluate(
        y_eval_pred_unc_u, y_eval_q10_unc_u, y_eval_q50_unc_u, y_eval_q90_unc_u,
        y_eval_std_pred_unc_u, y_eval_std_epi_unc_u, y_eval_std_ale_unc_u, "uncal"
    )

    # Post-calibrated (STD for std-metrics, quantile-calibrated q10/q50/q90) on EVAL
    results_cal = _evaluate(
        y_eval_pred_cal_u, y_eval_q10_qcal_u, y_eval_q50_qcal_u, y_eval_q90_qcal_u,
        y_eval_std_pred_cal_u, y_eval_std_epi_cal_u, y_eval_std_ale_cal_u, "cal"
    )

    # ---- SAVE ----
    os.makedirs("models/bnn", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    torch.save(model, "models/bnn/nba_bnn_full_model.pt")
    torch.save(model.state_dict(), "models/bnn/nba_bnn_weights_only.pt")
    # Post-calibrated (eval split) as the main table; uncalibrated (eval split) for reference
    results_cal.to_parquet("datasets/Evaluation_Metrics.parquet", index=False)
    results_uncal.to_parquet("datasets/Evaluation_Metrics_Uncal.parquet", index=False)
    # Save aux only if configured
    if model.aux_dim > 0:
        with torch.no_grad():
            aux_val_probs = model.forward_aux(
                X_val_tensor.to(device), X_val_embed_tensor.to(device)
            ).cpu().numpy()
        pd.DataFrame(aux_val_probs, columns=targets).to_parquet(
            "datasets/AuxBreakout_Probs.parquet", index=False
        )
    # Save both calibration scales
    pd.DataFrame({
        "Target": targets,
        "Aleatoric_Scale": calib_scales,
        "Quantile_Scale": q_scales,
    }).to_parquet("datasets/Calibration_Scales.parquet", index=False)

    artifacts = BNNArtifacts(
        model=model,
        best_state=best_model_state,
        y_scaler=y_scaler,
        targets=targets,
        train_index=pre_art.train_index,
        val_index=pre_art.val_index,
        embedding_sizes=pre_art.embedding_sizes,
        calib_scales=calib_scales,
    )
    return results_cal, artifacts
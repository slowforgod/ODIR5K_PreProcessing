"""
preprocessing/v2/losses.py

Loss functions for the V2 (Data Augmentation) experiment.

Losses:
    1. bce_with_logits_loss    — nn.BCEWithLogitsLoss()  (baseline)
    2. focal_loss              — multi-label Focal Loss (Lin et al. 2017)
    3. robust_asymmetric_loss  — ASL + Hill term (Ben-Baruch et al. 2020 variant)
    4. cb_bce_with_logits_loss — Class-Balanced BCE (Cui et al. 2019)
    5. cb_focal_loss           — Class-Balanced Focal

Factory:
    make_criterion(name, cb_weights, params) -> nn.Module

CRITICAL: CB weights (losses 4 & 5) MUST be pre-computed from the original
          train_df BEFORE LSE expansion and passed in as `cb_weights`.

Expected approximate CB weights (sanity check):
    N≈0.62, D≈0.62, G≈1.35, C≈1.27, A≈1.54, M≈1.55, H≈2.05
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# effective_number_class_weights
# ---------------------------------------------------------------------------

def effective_number_class_weights(
    labels_np: np.ndarray,
    beta: float = 0.9999,
    normalize: str = "mean_one",
    mode: str = "effective_number",
) -> torch.Tensor:
    """Compute class-balanced weights from per-class positive counts.

    Modes:
        "effective_number" (Cui et al. 2019):
            w[i] = (1 - beta) / (1 - beta ** n[i])
        "inverse_frequency":
            w[i] = 1 / n[i]      # purely proportional to count ratio

    Args:
        labels_np : (N, C) multi-hot float32 array — from ORIGINAL train_df
        beta      : smoothing factor for effective-number mode (default 0.9999)
        normalize : "mean_one" → scale so w.mean() == 1
        mode      : "effective_number" or "inverse_frequency"

    Returns:
        (C,) float32 torch.Tensor
    """
    pos_counts = labels_np.sum(axis=0).clip(min=1)    # (C,)

    if mode == "inverse_frequency":
        weights = 1.0 / pos_counts
    elif mode == "effective_number":
        weights = (1.0 - beta) / (1.0 - beta ** pos_counts)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'effective_number' or 'inverse_frequency'."
        )

    if normalize == "mean_one":
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-label Focal Loss (Lin et al. 2017).

    loss = -α_t · (1 - p_t)^γ · log(p_t)  per element, mean-reduced.

    α_t = α · y + (1 - α) · (1 - y)
    p_t = p · y + (1 - p) · (1 - y)    where p = sigmoid(logits)

    Args:
        alpha        : positive-class weight (default 0.75)
        gamma        : focusing exponent (default 2.0)
        class_weight : optional (C,) tensor; multiplied per-class if provided
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weight",
            class_weight if class_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) multi-hot float32

        Returns:
            scalar loss
        """
        eps = 1e-7
        p = torch.sigmoid(logits)

        # α_t: α for positives, (1-α) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # p_t: p for positives, (1-p) for negatives
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        # Binary cross-entropy term
        log_pt = torch.log(p_t.clamp(min=eps))

        loss = -alpha_t * focal_weight * log_pt   # (B, C)

        if self.class_weight is not None:
            loss = loss * self.class_weight.to(logits.device)

        return loss.mean()


# ---------------------------------------------------------------------------
# RobustAsymmetricLoss
# ---------------------------------------------------------------------------

class RobustAsymmetricLoss(nn.Module):
    """Robust Asymmetric Loss (ASL + Hill polynomial variant for negatives).

    Based on Ben-Baruch et al. 2020 Asymmetric Loss, with the negative-side
    log term replaced by a bounded polynomial (Hill-style) for label-noise
    robustness.

    Components:
        p      = sigmoid(logits)
        p_pos  = p
        p_neg  = clamp(1 - p + clip, max=1)    # asymmetric clipping for negatives

        # Positive side: standard log (with positive focusing in ASL form)
        loss_pos = -y · log(clamp(p_pos, min=ε))

        # Negative side: bounded polynomial — REPLACES -(1-y)·log(p_neg)
        #   loss_neg = (1-y) · ((1 - p_neg).clamp(min=0))^N / N
        # When (1 - p_neg) is small (model correctly low-prob), loss is small.
        # When (1 - p_neg) is large (model confidently wrong, possibly label noise),
        # the polynomial bounds the gradient instead of letting log blow up.
        # This is the "Hill" robustness intent: down-weight potentially-noisy negatives.

        # Asymmetric focusing (standard ASL) applied to both sides
        pt           = p_pos · y + p_neg · (1 - y)
        one_sided_γ  = γ_pos · y + γ_neg · (1 - y)
        focal_weight = (1 - pt)^one_sided_γ

        loss = (loss_pos + loss_neg) · focal_weight

    Notes:
        - `lam` (RAL_LAMBDA) and `pos_terms` are kept in config for forward
          compatibility with other ASL/Hill variants but are NOT used in this
          negative-only polynomial formulation. If a future variant uses them,
          extend this class accordingly.
        - The standard Hill loss in Ben-Baruch et al. is defined at the gradient
          level. The loss-level approximation here matches the noise-robust
          intent (bounded loss on confidently-wrong predictions).

    Args:
        gamma_neg  : focusing exponent for negatives (default 2.0)
        gamma_pos  : focusing exponent for positives (default 1.0)
        clip       : probability shift for negatives (default 0.05)
        lam        : kept for config compatibility; unused (default 1.5)
        pos_terms  : kept for config compatibility; unused (default 2)
        neg_terms  : polynomial degree for negative Hill term (default 2)
    """

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        lam: float = 1.5,
        pos_terms: int = 2,
        neg_terms: int = 2,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.lam = lam            # unused — kept for compatibility
        self.pos_terms = pos_terms  # unused — kept for compatibility
        self.neg_terms = neg_terms

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) multi-hot float32

        Returns:
            scalar loss
        """
        eps = 1e-7
        p = torch.sigmoid(logits)

        p_pos = p
        p_neg = (1.0 - p + self.clip).clamp(max=1.0)

        # Positive: standard log-form
        log_pos = torch.log(p_pos.clamp(min=eps))
        loss_pos = -targets * log_pos

        # Negative: Hill polynomial — bounded, noise-robust replacement for log(p_neg)
        neg_diff = (1.0 - p_neg).clamp(min=0)
        loss_neg = (1.0 - targets) * (neg_diff ** self.neg_terms) / self.neg_terms

        # Asymmetric focusing on both sides
        pt = p_pos * targets + p_neg * (1.0 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = (1.0 - pt) ** one_sided_gamma

        loss = (loss_pos + loss_neg) * focal_weight
        return loss.mean()


# ---------------------------------------------------------------------------
# CBBCEWithLogitsLoss
# ---------------------------------------------------------------------------

class CBBCEWithLogitsLoss(nn.Module):
    """Class-Balanced BCE with Logits Loss (Cui et al. 2019).

    CB weight is applied as an element-wise multiplier on the per-class
    binary cross-entropy loss:
        loss = class_weight[c] · BCE(logit_c, target_c)

    Args:
        class_weight : (C,) float32 tensor (pre-normalised effective-number weights)
    """

    def __init__(self, class_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) multi-hot float32

        Returns:
            scalar loss
        """
        # per-element BCE without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, C)
        weighted = bce * self.class_weight.to(logits.device)
        return weighted.mean()


# ---------------------------------------------------------------------------
# CBFocalLoss
# ---------------------------------------------------------------------------

class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss.

    Focal Loss (Lin et al. 2017) with CB weights (Cui et al. 2019).
    Delegates to FocalLoss with class_weight provided.

    Args:
        alpha        : positive-class weight (default 0.75)
        gamma        : focusing exponent (default 2.0)
        class_weight : (C,) float32 tensor (pre-normalised effective-number weights)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._focal = FocalLoss(alpha=alpha, gamma=gamma, class_weight=class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._focal(logits, targets)


# ---------------------------------------------------------------------------
# make_criterion
# ---------------------------------------------------------------------------

def make_criterion(
    name: str,
    cb_weights: Optional[torch.Tensor] = None,
    params: Optional[dict] = None,
) -> nn.Module:
    """Factory: return the appropriate loss nn.Module by name.

    Args:
        name       : one of the five V2 loss names
        cb_weights : (C,) float32 tensor from effective_number_class_weights()
                     Required for 'cb_bce_with_logits_loss' and 'cb_focal_loss'.
                     Must be computed from the ORIGINAL train_df (pre-LSE).
        params     : optional dict of hyperparameter overrides (from config).
                     Keys: focal_alpha, focal_gamma, ral_gamma_neg, ral_gamma_pos,
                           ral_clip, ral_lambda, ral_pos_terms, ral_neg_terms,
                           cb_beta, cb_weight_normalize.

    Returns:
        nn.Module with forward(logits, targets) -> scalar loss

    Raises:
        ValueError : for unknown loss names or missing cb_weights for CB losses.
    """
    if params is None:
        params = {}

    focal_alpha = float(params.get("focal_alpha", 0.75))
    focal_gamma = float(params.get("focal_gamma", 2.0))
    ral_gamma_neg = float(params.get("ral_gamma_neg", 2.0))
    ral_gamma_pos = float(params.get("ral_gamma_pos", 1.0))
    ral_clip = float(params.get("ral_clip", 0.05))
    ral_lambda = float(params.get("ral_lambda", 1.5))
    ral_pos_terms = int(params.get("ral_pos_terms", 2))
    ral_neg_terms = int(params.get("ral_neg_terms", 2))

    if name == "bce_with_logits_loss":
        return nn.BCEWithLogitsLoss()

    if name == "focal_loss":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    if name == "robust_asymmetric_loss":
        return RobustAsymmetricLoss(
            gamma_neg=ral_gamma_neg,
            gamma_pos=ral_gamma_pos,
            clip=ral_clip,
            lam=ral_lambda,
            pos_terms=ral_pos_terms,
            neg_terms=ral_neg_terms,
        )

    if name in ("cb_bce_with_logits_loss", "cb_focal_loss"):
        if cb_weights is None:
            raise ValueError(
                f"cb_weights must be provided for '{name}'. "
                "Compute from original train_df (pre-LSE) using "
                "effective_number_class_weights()."
            )

    if name == "cb_bce_with_logits_loss":
        return CBBCEWithLogitsLoss(class_weight=cb_weights)

    if name == "cb_focal_loss":
        return CBFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weight=cb_weights,
        )

    raise ValueError(
        f"Unknown loss name '{name}'. Valid names: "
        "bce_with_logits_loss, focal_loss, robust_asymmetric_loss, "
        "cb_bce_with_logits_loss, cb_focal_loss."
    )

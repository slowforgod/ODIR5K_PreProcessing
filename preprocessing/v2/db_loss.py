"""
preprocessing/v2/db_loss.py

Distribution-Balanced Loss (DB Loss) for multi-label classification.

Reference: wutong16/DistributionBalancedLoss (Wu et al., ECCV 2020,
"Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed
Datasets"). Follows the official `ResampleLoss` formulation.

Two components (both required):
  1) Re-balanced weighting (`_rebalance_weight`):
       freq_inv = 1 / class_freq
       repeat_rate_i = sum_c (target_ic * freq_inv_c)          # per sample
       pos_weight_ic = freq_inv_c / repeat_rate_i
       weight_ic     = sigmoid(beta * (pos_weight_ic - gamma)) + alpha
  2) Negative-tolerant regularization / logit reg (`_logit_reg`):
       init_bias_c = -log(num_samples / class_freq_c - 1) * init_bias
       logits += init_bias
       logits  = logits * (1-t) * neg_scale + logits * t       # scale neg branch
       weight  = weight / neg_scale * (1-t) + weight * t        # rescale neg weight
     Relieves over-suppression of negative labels for rare classes.

Class frequency (per-class positive count) + total sample count are computed
ONCE from the train split and injected via `class_freq` / `num_samples`.

Default hyperparameters are the COCO config values from the official repo
(map: alpha=0.1, beta=10, gamma=0.2 / logit_reg: neg_scale=2.0, init_bias=0.05).
NOTE: this is a 7-class fundus setting, not COCO's 80 classes — these are a
starting point and are expected to be TUNED. All are exposed via config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss (official ResampleLoss formulation).

    Args:
        class_freq  : (C,) per-class positive count from the ORIGINAL train split.
        num_samples : total number of training samples (int).
        map_alpha   : re-balance map smoothing offset (default 0.1).
        map_beta    : re-balance map slope            (default 10.0).
        map_gamma   : re-balance map shift            (default 0.2).
        neg_scale   : negative-branch logit/weight down-scale (default 2.0).
        init_bias   : per-class negative-branch logit bias coefficient (default 0.05).
    """

    def __init__(
        self,
        class_freq: torch.Tensor,
        num_samples: int,
        map_alpha: float = 0.1,
        map_beta: float = 10.0,
        map_gamma: float = 0.2,
        neg_scale: float = 2.0,
        init_bias: float = 0.05,
    ):
        super().__init__()
        eps = 1e-7

        class_freq = class_freq.float().clamp(min=eps)
        self.num_samples = int(num_samples)
        self.map_alpha = float(map_alpha)
        self.map_beta = float(map_beta)
        self.map_gamma = float(map_gamma)
        self.neg_scale = float(neg_scale)

        self.register_buffer("freq_inv", 1.0 / class_freq)        # (C,)

        # NTR per-class bias (official): -log(N/class_freq - 1) * init_bias
        bias = -torch.log(
            (float(num_samples) / class_freq) - 1.0 + eps
        ) * float(init_bias)
        self.register_buffer("init_bias", bias)                   # (C,)

    def _rebalance_weight(self, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-class re-balance weight. targets: (B, C) → (B, C)."""
        repeat_rate = (targets * self.freq_inv[None, :]).sum(dim=1, keepdim=True)  # (B,1)
        pos_weight = self.freq_inv[None, :] / repeat_rate.clamp(min=1e-7)          # (B,C)
        return torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw logits
            targets : (B, C) multi-hot float32
        Returns:
            scalar loss
        """
        targets = targets.float()
        weight = self._rebalance_weight(targets)                  # (B, C)

        # Negative-tolerant logit regularization
        bias = self.init_bias.to(logits.device)
        logits = logits + bias[None, :]
        logits = logits * (1.0 - targets) * self.neg_scale + logits * targets
        weight = weight / self.neg_scale * (1.0 - targets) + weight * targets

        cls_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (cls_loss * weight).mean()

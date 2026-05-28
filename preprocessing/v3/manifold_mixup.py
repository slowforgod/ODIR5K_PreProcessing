"""
preprocessing/v3/manifold_mixup.py

V3Preprocessor — Manifold Mixup at the feature level.
Called from the training loop (not the Dataset), so the signature differs
from all other V-preprocessors.  See preprocessing/v3/README.md for the
full algorithm spec.
"""

import numpy as np
import torch


class V3Preprocessor:
    """
    Manifold Mixup helper.

    NOTE: BasePreprocessor 와 시그니처가 다릅니다.
    학습 루프(train/train_v3.py)에서 직접 호출되며,
    Dataset 단계가 아니라 모델 forward 도중에 끼어듭니다.

    Args:
        alpha       : Beta(alpha, alpha) 파라미터 (기본 0.2)
        mixup_layers: 보간 후보 레이어 목록 (기본 [0, 1, 2, 3])
                      0 = pixel-level, 1~3 = ResNet layer{1..3} 직후
    """

    def __init__(self, alpha: float = 0.2, mixup_layers: list = None):
        if mixup_layers is None:
            mixup_layers = [0, 1, 2, 3]
        self.alpha = alpha
        self.mixup_layers = mixup_layers

    # ------------------------------------------------------------------
    # Core API (contracts from preprocessing/v3/README.md)
    # ------------------------------------------------------------------

    def apply(
        self,
        feat: torch.Tensor,
        label: torch.Tensor,
        alpha: float,
    ) -> tuple:
        """Manifold Mixup: 미니 배치 내 두 샘플을 feature 공간에서 보간.

        Args:
            feat  : (B, ...) — 섞을 feature tensor
            label : (B, num_classes) — multi-hot label tensor
            alpha : Beta(alpha, alpha) 파라미터

        Returns:
            mixed_feat : (B, ...) — lam * feat + (1-lam) * feat[perm]
            label_a    : (B, num_classes) — 원본 label
            label_b    : (B, num_classes) — 셔플된 label  (label[perm])
            lam        : float — Beta(alpha, alpha) 샘플 값
        """
        lam = self.sample_lam(alpha)
        perm = torch.randperm(feat.size(0), device=feat.device)

        # feature 보간 — feat 텐서의 나머지 차원(C, H, W 등)에 broadcast
        mixed_feat = lam * feat + (1.0 - lam) * feat[perm]

        label_a = label
        label_b = label[perm]

        return mixed_feat, label_a, label_b, lam

    # ------------------------------------------------------------------
    # Helper methods (used by ResNet50ManifoldMixup)
    # ------------------------------------------------------------------

    def sample_lam(self, alpha: float) -> float:
        """Beta(alpha, alpha) 에서 lambda 샘플링."""
        if alpha > 0:
            return float(np.random.beta(alpha, alpha))
        return 1.0

    def sample_layer(self, layers: list = None) -> int:
        """레이어 목록에서 한 개 무작위 선택."""
        if layers is None:
            layers = self.mixup_layers
        return int(np.random.choice(layers))

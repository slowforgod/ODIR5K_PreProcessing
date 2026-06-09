"""
model/efficientnet_b3.py

EfficientNet-B3 backbone with Input Mixup (pixel-space) support.

Architecture:
  torchvision EfficientNet-B3 features → avgpool → Dropout → Linear(1536, 7)

Mixup:
  Only pixel-space (k=0) input mixup is supported — EfficientNet's stage
  structure does not map cleanly onto ResNet layer indices, and the
  robustness experiment for V6 was scoped to input mixup only.
  Any non-zero k in `mixup_layers` is treated as k=0 (forced to input space).

forward signature follows model/README.md contract:
  model(x)                                         → logits  (B, 7)
  model(x, y=labels, alpha=a, mixup_layers=[...])  → (logits, label_a, label_b, lam)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from preprocessing.v3.manifold_mixup import V3Preprocessor


class EfficientNetB3InputMixup(nn.Module):
    """EfficientNet-B3 with optional pixel-space Mixup during training.

    Args:
        num_classes : number of output classes (default 7)
        pretrained  : load ImageNet weights from torchvision (default True)
        dropout     : Dropout rate before final FC (default 0.4 —
                      bumped from EfficientNet-B3's default 0.3 to suppress
                      overfitting on this small dataset)
        alpha       : default Beta(alpha, alpha) parameter for Mixup
        mixup_layers: kept for interface compatibility — effectively forced
                      to [0] (input mixup only)
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.4,
        alpha: float = 0.2,
        mixup_layers: list = None,
    ):
        super().__init__()

        weights = (
            models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base = models.efficientnet_b3(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        # EfficientNet-B3: penultimate feature dim = 1536.
        # Replace only the final Linear; raise the Dropout rate to `dropout`.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1536, num_classes),
        )

        self._mixup = V3Preprocessor(alpha=alpha, mixup_layers=[0])

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.avgpool(h)
        return torch.flatten(h, 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        alpha: float = 0.0,
        mixup_layers: list = None,
    ):
        """
        Args:
            x            : (B, 3, 300, 300) input images
            y            : (B, 7) multi-hot labels — None for plain inference
            alpha        : Beta(alpha, alpha) parameter (0 → no Mixup)
            mixup_layers : ignored — input mixup is the only supported mode.

        Returns:
            plain inference / no y / not training:
                logits — (B, 7)
            training with y and alpha > 0:
                (logits, label_a, label_b, lam)
        """
        if y is None or not self.training or alpha == 0.0:
            return self.classifier(self._encode(x))

        mixed_x, label_a, label_b, lam = self._mixup.apply(x, y, alpha)
        logits = self.classifier(self._encode(mixed_x))
        return logits, label_a, label_b, lam

"""
model/resnet50.py

ResNet-50 backbone with Manifold Mixup support.

Architecture:
  stem → layer1 → layer2 → layer3 → layer4 → avgpool → Dropout → FC(7)

Mixup insertion points (k):
  k=0  pixel space (standard Input Mixup)
  k=1  after layer1   56×56 / 256ch
  k=2  after layer2   28×28 / 512ch
  k=3  after layer3   14×14 / 1024ch

forward signature follows model/README.md:
  model(x)                                 → logits  (B, 7)
  model(x, y=labels, alpha=a, mixup_layers=[...]) → (logits, label_a, label_b, lam)
"""

import random

import torch
import torch.nn as nn
import torchvision.models as models

from preprocessing.v3.manifold_mixup import V3Preprocessor


class ResNet50ManifoldMixup(nn.Module):
    """ResNet-50 with optional Manifold Mixup during training.

    Args:
        num_classes : number of output classes (default 7)
        pretrained  : load ImageNet weights from torchvision (default True)
        dropout     : Dropout rate before final FC (default 0.5)
        alpha       : default Beta(alpha, alpha) parameter for Mixup
        mixup_layers: default candidate layers for Mixup
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5,
        alpha: float = 0.2,
        mixup_layers: list = None,
    ):
        super().__init__()

        if mixup_layers is None:
            mixup_layers = [0, 1, 2, 3]

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

        # Split backbone into named segments for selective forwarding
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Classification head: Dropout → FC
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

        # V3Preprocessor handles all Mixup math
        self._mixup = V3Preprocessor(alpha=alpha, mixup_layers=mixup_layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_until(self, x: torch.Tensor, until: int) -> torch.Tensor:
        """Forward x through stem + layers 1..until (inclusive)."""
        h = self.stem(x)
        if until >= 1:
            h = self.layer1(h)
        if until >= 2:
            h = self.layer2(h)
        if until >= 3:
            h = self.layer3(h)
        return h

    def _decode_from(self, h: torch.Tensor, from_: int) -> torch.Tensor:
        """Forward feature h starting from the layer *after* from_."""
        if from_ < 2:
            h = self.layer2(h)
        if from_ < 3:
            h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        return self.classifier(h)

    # ------------------------------------------------------------------
    # forward  — contract from model/README.md
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        alpha: float = 0.0,
        mixup_layers: list = None,
    ):
        """
        Args:
            x            : (B, 3, H, W) input images
            y            : (B, 7) multi-hot labels — None for plain inference
            alpha        : Beta(alpha, alpha) parameter (0 → no Mixup)
            mixup_layers : candidate layers to insert Mixup

        Returns:
            y is None or not training:
                logits — (B, 7)
            y given and self.training:
                (logits, label_a, label_b, lam)
        """
        # ----- plain forward (inference / val) -----
        if y is None or not self.training or alpha == 0.0:
            h = self._encode_until(x, until=3)
            return self._decode_from(h, from_=3)

        # ----- Manifold Mixup forward (training) -----
        if mixup_layers is None:
            mixup_layers = self._mixup.mixup_layers

        # Pick one layer uniformly each forward pass
        k = self._mixup.sample_layer(mixup_layers)

        if k == 0:
            # Pixel-space mixup: mix inputs, then full forward
            mixed_x, label_a, label_b, lam = self._mixup.apply(x, y, alpha)
            h = self._encode_until(mixed_x, until=3)
            logits = self._decode_from(h, from_=3)
        else:
            # Feature-space mixup: encode both originals up to layer k,
            # mix features, then decode the rest
            h = self._encode_until(x, until=k)
            mixed_h, label_a, label_b, lam = self._mixup.apply(h, y, alpha)
            logits = self._decode_from(mixed_h, from_=k)

        return logits, label_a, label_b, lam

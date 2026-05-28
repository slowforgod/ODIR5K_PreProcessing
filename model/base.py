"""
model/base.py

Factory function for all backbone models.
train/ code always uses build_model() — never imports model classes directly.
"""

import torch.nn as nn


def build_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """Build and return a model by name.

    Args:
        name        : "resnet50" | "densenet" | "efficientnet"
        num_classes : output classes (default 7: N D G C A H M)
        pretrained  : load ImageNet weights (default True)
        dropout     : Dropout rate before final FC (default 0.5)

    Returns:
        nn.Module with forward matching model/README.md contract.

    Raises:
        NotImplementedError: for densenet / efficientnet (other teammates).
        ValueError         : for unknown model names.
    """
    name = name.lower().strip()

    if name == "resnet50":
        from model.resnet50 import ResNet50ManifoldMixup
        return ResNet50ManifoldMixup(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if name in ("densenet", "efficientnet"):
        raise NotImplementedError(
            f"V3 only requires ResNet-50; '{name}' is for other teammates' Vs."
        )

    raise ValueError(
        f"Unknown model name '{name}'. Supported: 'resnet50', 'densenet', 'efficientnet'."
    )

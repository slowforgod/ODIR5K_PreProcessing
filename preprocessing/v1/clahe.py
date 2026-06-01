"""
preprocessing/v1/clahe.py

V1 Preprocessor: Selectable CLAHE variant on LAB colour space channels.

Variants:
    V0      : identity passthrough (no CLAHE)
    V1_L    : CLAHE on L channel only  (original V1)
    V1_LA   : CLAHE on L + A channels
    V1_LB   : CLAHE on L + B channels
    V1_LAB  : CLAHE on L + A + B channels

Algorithm (non-V0):
    RGB → LAB (cv2.COLOR_RGB2LAB) → CLAHE on selected channels → LAB → RGB

CLAHE object is created once in __init__ and reused across all apply() calls.
"""

import cv2
import numpy as np


_VARIANT_CHANNELS = {
    "V1_L":   [0],
    "V1_LA":  [0, 1],
    "V1_LB":  [0, 2],
    "V1_LAB": [0, 1, 2],
}

_VALID_VARIANTS = {"V0", "V1_L", "V1_LA", "V1_LB", "V1_LAB"}


class V1Preprocessor:
    """CLAHE variant selectable via ``variant`` parameter.

    Args:
        variant        : one of "V0", "V1_L", "V1_LA", "V1_LB", "V1_LAB"
        clip_limit     : CLAHE clipLimit (default 2.0)
        tile_grid_size : CLAHE tileGridSize (default (8, 8))
    """

    def __init__(
        self,
        variant: str = "V1_L",
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ):
        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {sorted(_VALID_VARIANTS)}, got '{variant}'"
            )
        self.variant = variant
        self._channels = _VARIANT_CHANNELS.get(variant)  # None for V0

        # Create CLAHE object once; reused across all apply() calls.
        if self._channels is not None:
            self._clahe = cv2.createCLAHE(
                clipLimit=float(clip_limit),
                tileGridSize=tuple(tile_grid_size),
            )
        else:
            self._clahe = None

    def apply(self, image: np.ndarray, label=None):
        """Apply the configured CLAHE variant to a single image.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 or None — passed through unchanged

        Returns:
            (result_image, label)
            result_image : (H, W, 3) RGB uint8 numpy array
        """
        if self._channels is None:
            # V0: identity passthrough — no copy needed
            return image, label

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Split into individual planes in-place via indexing (avoids cv2.split alloc)
        for ch in self._channels:
            lab[:, :, ch] = self._clahe.apply(lab[:, :, ch])

        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result, label

    def is_train_only(self) -> bool:
        """False — CLAHE applied to both train and val loaders."""
        return False

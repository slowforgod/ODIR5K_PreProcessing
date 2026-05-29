"""
preprocessing/v1/clahe.py

V1 Preprocessor: CLAHE on L channel of LAB colour space.

Algorithm:
    1. RGB → LAB  (cv2.COLOR_RGB2LAB)
    2. Split into L, A, B channels
    3. Apply CLAHE to L channel only  (A, B untouched — no colour distortion)
    4. Merge channels back
    5. LAB → RGB  (cv2.COLOR_LAB2RGB)

Parameters (from configs/v1.yaml):
    clip_limit     : CLAHE clip limit          (default 2.0)
    tile_grid_size : CLAHE tile grid size tuple (default (8, 8))
"""

import cv2
import numpy as np


class V1Preprocessor:
    """CLAHE on L channel only.

    Args:
        clip_limit     : passed to cv2.createCLAHE (default 2.0)
        tile_grid_size : passed to cv2.createCLAHE (default (8, 8))
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, image: np.ndarray, label: np.ndarray = None):
        """Apply CLAHE to L channel.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32. Passed through unchanged.

        Returns:
            (clahe_image, label)
            clahe_image : (H, W, 3) RGB uint8 numpy array
        """
        # RGB → LAB (OpenCV expects BGR input, but COLOR_RGB2LAB handles RGB directly)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L channel only
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        l_clahe = clahe.apply(l)

        # Merge and convert back to RGB
        lab_clahe = cv2.merge((l_clahe, a, b))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return result, label

    def is_train_only(self) -> bool:
        """False — apply to both train and val loaders."""
        return False
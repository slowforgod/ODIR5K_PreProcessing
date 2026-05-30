"""
preprocessing/v1/clahe_B.py

V1 Preprocessor: CLAHE on B channel only of LAB colour space.

Algorithm:
    1. RGB → LAB  (cv2.COLOR_RGB2LAB)
    2. Split into L, A, B channels
    3. Apply CLAHE to B channel only  (L, A untouched)
       - B: blue↔yellow colour contrast enhancement → macula/hard exudate visibility
    4. Merge channels back
    5. LAB → RGB  (cv2.COLOR_LAB2RGB)

Parameters (from configs/v1_B.yaml):
    clip_limit     : CLAHE clip limit          (default 2.0)
    tile_grid_size : CLAHE tile grid size tuple (default (8, 8))
"""

import cv2
import numpy as np


class V1BPreprocessor:
    """CLAHE on B channel only.

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
        """Apply CLAHE to B channel only.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32. Passed through unchanged.

        Returns:
            (clahe_image, label)
            clahe_image : (H, W, 3) RGB uint8 numpy array
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        b_clahe = clahe.apply(b)

        lab_clahe = cv2.merge((l, a, b_clahe))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return result, label

    def is_train_only(self) -> bool:
        """False — apply to both train and val loaders."""
        return False
"""
preprocessing/v4/preprocess.py

V4Preprocessor: V1 (CLAHE) + V2 (augmentation) combined dataset-side preprocessor.

Apply order:
    1. V1Preprocessor.apply()  — CLAHE on LAB channels
    2. V2Preprocessor.apply()  — albumentations augmentation

This preprocessor is training-only (is_train_only → True) because V2
augmentation must not be applied to the validation set.
"""

from preprocessing.v1.clahe import V1Preprocessor
from preprocessing.v2.augmentation import V2Preprocessor


class V4Preprocessor:
    """Dataset-side V1 + V2 composite preprocessor.

    Args:
        v1_kwargs : keyword arguments forwarded to V1Preprocessor
                    (variant, clip_limit, tile_grid_size)
        v2_kwargs : keyword arguments forwarded to V2Preprocessor
                    (img_size, hflip_p, vflip_p, geo_p, ...)
    """

    def __init__(self, v1_kwargs: dict, v2_kwargs: dict):
        self.v1 = V1Preprocessor(**v1_kwargs)
        self.v2 = V2Preprocessor(**v2_kwargs)

    def apply(self, image, label=None):
        """Apply V1 then V2 augmentation to a single image.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 or None — passed through unchanged

        Returns:
            (transformed_image, label)
        """
        image, label = self.v1.apply(image, label)
        image, label = self.v2.apply(image, label)
        return image, label

    def is_train_only(self) -> bool:
        """True — V2 augmentation must NOT be applied to the validation set."""
        return True

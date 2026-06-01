"""
preprocessing/v6/preprocess.py

V6Preprocessor: V1 (CLAHE) + V2 (augmentation) combined dataset-side preprocessor.
V3 Manifold Mixup is handled by the trainer, not the dataset.
Identical to V4Preprocessor in structure. Both apply V1 then V2 in sequence,
and both are training-only (val set uses V1 only, no V2 augmentation).
"""
from preprocessing.v1.clahe import V1Preprocessor
from preprocessing.v2.augmentation import V2Preprocessor


class V6Preprocessor:
    """Dataset-side V1 + V2 composite preprocessor for V6 experiments.

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
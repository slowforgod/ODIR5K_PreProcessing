"""
preprocessing/v5/preprocess.py

V5Preprocessor: V1 (CLAHE) only, dataset-side.
Manifold Mixup (V3) is handled by the trainer, not the dataset.

Because CLAHE is a deterministic enhancement (not random augmentation),
it is applied to both train and val sets → is_train_only() → False.
"""

from preprocessing.v1.clahe import V1Preprocessor


class V5Preprocessor:
    """Dataset-side V1 CLAHE preprocessor for V5 experiments.

    Args:
        v1_kwargs : keyword arguments forwarded to V1Preprocessor
                    (variant, clip_limit, tile_grid_size)
    """

    def __init__(self, v1_kwargs: dict):
        self.v1 = V1Preprocessor(**v1_kwargs)

    def apply(self, image, label=None):
        """Apply CLAHE variant to a single image.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 or None — passed through unchanged

        Returns:
            (transformed_image, label)
        """
        return self.v1.apply(image, label)

    def is_train_only(self) -> bool:
        """False — CLAHE is applied to both train and val sets."""
        return False

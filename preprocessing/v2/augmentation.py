"""
preprocessing/v2/augmentation.py

V2Preprocessor — albumentations-based data augmentation for fundus images.

Augmentation pipeline (applied only during training):
    Resize(224, 224)
    HorizontalFlip(p)
    VerticalFlip(p)
    OneOf([Rotate, Affine, RandomResizedCrop], p=geo_p)
    ColorJitter(p)         ← OUTSIDE OneOf (applied independently)
    GaussianBlur(p)        ← OUTSIDE OneOf (applied independently)

All probabilities and limits are constructor kwargs, readable from config.
"""

import numpy as np
import albumentations as A


class V2Preprocessor:
    """Albumentations augmentation pipeline for V2.

    Args:
        img_size           : output square size (default 224)
        hflip_p            : HorizontalFlip probability
        vflip_p            : VerticalFlip probability
        geo_p              : OneOf geometric group probability
        rotate_limit       : max rotation degrees for Rotate
        affine_scale       : (min, max) scale range for Affine
        affine_translate   : max translate_percent for Affine
        affine_rotate      : max rotate degrees for Affine (symmetric ±)
        affine_shear       : max shear degrees for Affine (symmetric ±)
        crop_scale         : (min, max) scale range for RandomResizedCrop
        color_jitter_p     : ColorJitter probability
        brightness         : brightness jitter limit
        contrast           : contrast jitter limit
        saturation         : saturation jitter limit
        blur_p             : GaussianBlur probability
        blur_limit         : (min, max) blur kernel size (odd ints)
    """

    def __init__(
        self,
        img_size: int = 224,
        hflip_p: float = 0.5,
        vflip_p: float = 0.1,
        geo_p: float = 0.8,
        rotate_limit: int = 20,
        affine_scale: tuple = (0.85, 1.15),
        affine_translate: float = 0.07,
        affine_rotate: int = 15,
        affine_shear: int = 5,
        crop_scale: tuple = (0.82, 1.0),
        color_jitter_p: float = 0.3,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.05,
        blur_p: float = 0.2,
        blur_limit: tuple = (3, 5),
        **kwargs,
    ):
        # Coerce list→tuple for YAML-loaded sequences
        affine_scale = tuple(affine_scale)
        crop_scale = tuple(crop_scale)
        blur_limit = tuple(blur_limit)

        self._transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=hflip_p),
            A.VerticalFlip(p=vflip_p),
            A.OneOf([
                A.Rotate(limit=rotate_limit, p=1.0),
                A.Affine(
                    scale=affine_scale,
                    translate_percent=affine_translate,
                    rotate=(-affine_rotate, affine_rotate),
                    # Explicit: x-axis only (fundus y-shear distorts disc/vessel structure unnaturally)
                    shear={"x": (-affine_shear, affine_shear), "y": (0, 0)},
                    p=1.0,
                ),
                # albumentations 1.4+ API: size=(h, w)
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=crop_scale,
                    p=1.0,
                ),
            ], p=geo_p),
            # ColorJitter and GaussianBlur are OUTSIDE OneOf — each applied independently
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=0,          # hue=0 intentional: fundus colour carries diagnostic signal
                p=color_jitter_p,
            ),
            A.GaussianBlur(blur_limit=blur_limit, p=blur_p),
        ])

    def apply(self, image: np.ndarray, label=None):
        """Apply augmentation to a single RGB uint8 image.

        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 array or None — passed through unchanged

        Returns:
            (transformed_image, label)
            transformed_image : (H, W, 3) RGB uint8 numpy array
        """
        result = self._transform(image=image)
        return result["image"], label

    def is_train_only(self) -> bool:
        """Augmentation is applied only during training."""
        return True

"""
preprocessing/common/crop.py

Black-border cropping for fundus images.

ODIR-5K fundus images are circular discs on a near-black square canvas.
crop_black_border() tightens the bounding box to the disc so that downstream
CLAHE / resize / augmentation operate on the informative region only.

Idempotent: applying it to an already-cropped image is (within a few pixels)
a no-op.
"""

import cv2
import numpy as np


def crop_black_border(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the black square border around a fundus disc.

    Args:
        image : (H, W, 3) RGB uint8 numpy array
        tol   : grayscale pixel value treated as "still black" (default 7).

    Returns:
        cropped (H', W', 3) RGB uint8 array. If no non-black region is found
        the input image is returned unchanged.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if not mask.any():
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    return image[y0:y1 + 1, x0:x1 + 1]

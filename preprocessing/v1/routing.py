"""
preprocessing/v1/routing.py

Per-class routing utilities for the V1 multi-variant CLAHE experiment.

Three routing modes:
    hard_route          : each class column comes from its designated variant
    soft_route_uniform  : uniform average across all variants, per class
    auto_route_from_val : per-class best-variant map derived from val F1
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score


DEFAULT_HARD_ROUTING: Dict[str, str] = {
    "N": "V1_LA",
    "D": "V1_LB",
    "G": "V0",
    "C": "V1_LAB",
    "A": "V1_L",
    "H": "V1_LB",
    "M": "V1_L",
}

DEFAULT_VARIANTS: List[str] = ["V0", "V1_L", "V1_LA", "V1_LB", "V1_LAB"]


def hard_route(
    probs_by_variant: Dict[str, np.ndarray],
    class_names: List[str],
    routing_map: Dict[str, str],
) -> np.ndarray:
    """For each class column, pick the assigned variant's probability column.

    Args:
        probs_by_variant : dict mapping variant name → (N, C) ndarray
        class_names      : list of C class name strings (column order)
        routing_map      : dict mapping class name → variant name

    Returns:
        (N, C) ndarray with each column sourced from its assigned variant

    Raises:
        KeyError: if a variant referenced in routing_map is absent from
                  probs_by_variant
    """
    missing = set(routing_map.values()) - set(probs_by_variant.keys())
    if missing:
        raise KeyError(
            f"hard_route: variants missing from probs_by_variant: {sorted(missing)}"
        )

    n = next(iter(probs_by_variant.values())).shape[0]
    c = len(class_names)
    out = np.empty((n, c), dtype=np.float32)

    for col, cls in enumerate(class_names):
        variant = routing_map[cls]
        out[:, col] = probs_by_variant[variant][:, col]

    return out


def soft_route_uniform(
    probs_by_variant: Dict[str, np.ndarray],
    class_names: List[str],
) -> np.ndarray:
    """Uniform average of all variant probability matrices.

    Uses np.stack + np.mean for a single vectorised reduction — no Python
    loop over rows or columns.

    Args:
        probs_by_variant : dict mapping variant name → (N, C) ndarray
        class_names      : list of C class name strings (unused beyond doc)

    Returns:
        (N, C) ndarray — element-wise mean across all variants
    """
    # stack → (V, N, C), mean over axis=0 → (N, C)
    stacked = np.stack(list(probs_by_variant.values()), axis=0)  # (V, N, C)
    return stacked.mean(axis=0).astype(np.float32)


def auto_route_from_val(
    probs_by_variant: Dict[str, np.ndarray],
    labels_val: np.ndarray,
    class_names: List[str],
    thresholds_by_variant: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, str]:
    """Pick the best variant per class based on val per-class F1.

    For each class column c:
        For each variant v, binarise probs_by_variant[v][:, c] with its own
        threshold (thresholds_by_variant[v][c]; defaults to 0.5), then compute
        F1 vs labels_val[:, c]. The variant with the highest F1 wins.

    The returned dict can be passed straight to `hard_route`.

    Args:
        probs_by_variant       : dict variant → (N, C) val probs
        labels_val             : (N, C) val multi-hot labels
        class_names            : list of C class names (column order)
        thresholds_by_variant  : optional dict variant → (C,) thresholds.
                                 If None or a variant key missing, falls back
                                 to 0.5 for that variant.

    Returns:
        routing_map : dict class_name → variant_name
    """
    if not probs_by_variant:
        raise ValueError("auto_route_from_val: probs_by_variant is empty")

    routing_map: Dict[str, str] = {}
    for col, cls in enumerate(class_names):
        best_variant, best_f1 = None, -1.0
        for variant, probs in probs_by_variant.items():
            if thresholds_by_variant and variant in thresholds_by_variant:
                thr = float(thresholds_by_variant[variant][col])
            else:
                thr = 0.5
            pred = (probs[:, col] >= thr).astype(int)
            f1c = f1_score(labels_val[:, col], pred, zero_division=0)
            if f1c > best_f1:
                best_f1, best_variant = f1c, variant
        routing_map[cls] = best_variant
    return routing_map

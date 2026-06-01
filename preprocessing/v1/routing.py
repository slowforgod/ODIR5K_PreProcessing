"""
preprocessing/v1/routing.py

Per-class routing utilities for the V1 multi-variant CLAHE experiment.

Two routing modes:
    hard_route        : each class column comes from its designated variant
    soft_route_uniform: uniform average across all variants, per class
"""

from typing import Dict, List

import numpy as np


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

"""
preprocessing/v2/oversampling.py

LSE (Label-Space Expansion) oversampling at the training dataset level.

Algorithm: greedy per-class expansion.
For each class c with positive count < target_count:
  1. Collect source rows where class c is positive.
  2. Shuffle source rows deterministically with the provided seed.
  3. Cycle through sources, adding one augmented copy per iteration.
     Each source row's augmented copy count is capped at max_aug_per_source.
  4. Stop when target_count is reached OR all sources have hit the cap.
  5. Multi-label: adding a copy of row R simultaneously increments counts
     for ALL classes that R has positive (greedy cross-class bookkeeping).

This function does NOT touch the validation set.
"""

import random
from typing import List

import numpy as np
import pandas as pd


def expand_with_lse(
    train_df: pd.DataFrame,
    class_cols: list,
    target_count: int,
    max_aug_per_source: int,
    seed: int,
    per_class_target: dict = None,
    per_class_max_aug: dict = None,
) -> List[dict]:
    """Expand the training set using greedy per-class LSE oversampling.

    Args:
        train_df           : original training DataFrame (pre-expansion)
        class_cols         : list of 7 class column names (e.g. ['N','D','G','C','A','H','M'])
        target_count       : default desired positive count per class after expansion
        max_aug_per_source : default per-source-row cap on augmented copies
        seed               : RNG seed for deterministic shuffling
        per_class_target   : optional {class_name: int} overriding target_count for
                             specific classes (e.g. {"H": 1100}). Classes absent from
                             the dict fall back to the scalar target_count.
        per_class_max_aug  : optional {class_name: int} overriding max_aug_per_source
                             for specific classes (e.g. {"H": 18}).

    Returns:
        List of record dicts:
            {"filename": str, "labels": np.ndarray(7,), "aug_idx": int}
        aug_idx == 0 for original rows; >= 1 for augmented copies.
        Every training sample still passes through V2Preprocessor at load time.
    """
    rng = random.Random(seed)

    labels_np = train_df[class_cols].values.astype(np.float32)   # (N, C)
    filenames = train_df["filename"].values
    num_classes = len(class_cols)
    n_orig = len(train_df)

    per_class_target = per_class_target or {}
    per_class_max_aug = per_class_max_aug or {}
    # Per-class resolved targets / caps (fall back to scalar defaults)
    targets = np.array(
        [int(per_class_target.get(c, target_count)) for c in class_cols], dtype=int
    )
    max_augs = np.array(
        [int(per_class_max_aug.get(c, max_aug_per_source)) for c in class_cols], dtype=int
    )

    # ------------------------------------------------------------------ #
    # Build initial records from the original rows                        #
    # ------------------------------------------------------------------ #
    records: List[dict] = [
        {
            "filename": str(filenames[i]),
            "labels": labels_np[i].copy(),
            "aug_idx": 0,
        }
        for i in range(n_orig)
    ]

    # Track current positive counts per class (will update as we add copies)
    pos_counts = labels_np.sum(axis=0).astype(int)   # (C,)

    # Track how many augmented copies each source row has produced
    aug_copy_count = np.zeros(n_orig, dtype=int)

    # ------------------------------------------------------------------ #
    # Greedy per-class expansion                                          #
    # ------------------------------------------------------------------ #
    print("\n[LSE] Oversampling summary:")
    print(f"  default target_count={target_count}  default max_aug_per_source={max_aug_per_source}")
    if per_class_target or per_class_max_aug:
        print(f"  per-class target overrides : {per_class_target}")
        print(f"  per-class max_aug overrides: {per_class_max_aug}")
    print(f"  Before expansion: total rows = {n_orig}")
    for ci, cls in enumerate(class_cols):
        print(f"    {cls}: {int(pos_counts[ci])}  (target={targets[ci]}, max_aug={max_augs[ci]})")

    for ci, cls in enumerate(class_cols):
        cls_target  = int(targets[ci])
        cls_max_aug = int(max_augs[ci])
        if pos_counts[ci] >= cls_target:
            continue  # already at or above target — skip

        # Source rows: original indices where class ci is positive
        source_indices = [i for i in range(n_orig) if labels_np[i, ci] == 1.0]
        if not source_indices:
            continue

        # Shuffle sources for this class
        rng.shuffle(source_indices)

        needed = cls_target - pos_counts[ci]
        added = 0
        cycle_pos = 0

        while added < needed:
            # Check if any source still has capacity
            if all(aug_copy_count[i] >= cls_max_aug for i in source_indices):
                break  # all sources capped — cannot reach target

            src_idx = source_indices[cycle_pos % len(source_indices)]
            cycle_pos += 1

            if aug_copy_count[src_idx] >= cls_max_aug:
                continue

            # Add one augmented copy of this source row
            aug_copy_count[src_idx] += 1
            copy_labels = labels_np[src_idx].copy()

            records.append({
                "filename": str(filenames[src_idx]),
                "labels": copy_labels,
                "aug_idx": aug_copy_count[src_idx],
            })

            # Update positive counts for ALL classes this row contributes to
            pos_counts += copy_labels.astype(int)
            added += 1

    print(f"\n  After expansion: total rows = {len(records)}")
    for ci, cls in enumerate(class_cols):
        print(f"    {cls}: {int(pos_counts[ci])}")
    print()

    return records

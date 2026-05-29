"""
analysis/metrics.py

Single source of truth for all evaluation metrics.
Every V-experiment must use these functions — do NOT compute metrics elsewhere.

Functions:
    find_optimal_thresholds(probs, labels, num_classes)
    compute_metrics(y_true, y_pred_proba, thresholds, class_names)
    save_metrics(metrics, save_dir, experiment_name)
    evaluate_with_tta(model, loader, device, num_classes, thresholds=None, class_names=None)
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# find_optimal_thresholds
# ---------------------------------------------------------------------------

def find_optimal_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Find per-class F1-maximising decision thresholds.

    Search range: np.arange(0.05, 0.96, 0.01) (0.05 ~ 0.95, step 0.01).
    If a class has no positive samples the threshold defaults to 0.5.

    Args:
        probs      : (N, num_classes) sigmoid probabilities
        labels     : (N, num_classes) multi-hot ground truth
        num_classes: number of classes

    Returns:
        thresholds : (num_classes,) float32 array
    """
    thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    candidates = np.arange(0.05, 0.96, 0.01)

    for c in range(num_classes):
        if labels[:, c].sum() == 0:
            # No positives — keep 0.5 and skip
            continue
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            pred_c = (probs[:, c] >= t).astype(int)
            f1c = f1_score(labels[:, c], pred_c, zero_division=0)
            if f1c > best_f1:
                best_f1, best_t = f1c, float(t)
        thresholds[c] = best_t

    return thresholds


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray,
    class_names: list,
) -> dict:
    """Compute all 5 evaluation metrics from analysis/README.md.

    Args:
        y_true       : (N, C) multi-hot ground truth
        y_pred_proba : (N, C) sigmoid probabilities
        thresholds   : (C,) decision thresholds per class
        class_names  : list of C class name strings

    Returns:
        dict with keys: macro_auc, per_class_auc, macro_f1, per_class_f1,
                        kappa, per_class_kappa, sensitivity, specificity
    """
    num_classes = len(class_names)

    # Binarise predictions using per-class thresholds
    y_pred_binary = (y_pred_proba >= thresholds[None, :]).astype(int)

    # ---- macro_auc ----
    try:
        macro_auc = float(roc_auc_score(y_true, y_pred_proba, average="macro"))
    except ValueError:
        macro_auc = float("nan")

    # ---- per_class_auc ----
    per_class_auc = {}
    for i, cls in enumerate(class_names):
        pos = int(y_true[:, i].sum())
        if 0 < pos < len(y_true):
            per_class_auc[cls] = float(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
        else:
            per_class_auc[cls] = None

    # ---- macro_f1 + per_class_f1 ----
    macro_f1 = float(
        f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    )
    per_class_f1_arr = f1_score(
        y_true, y_pred_binary, average=None, zero_division=0
    )
    per_class_f1 = {
        cls: float(per_class_f1_arr[i]) for i, cls in enumerate(class_names)
    }

    # ---- kappa (per-class quadratic-weighted, averaged over valid classes) ----
    # per_class_kappa[c] is None if c has only one class present (kappa undefined).
    per_class_kappa = {}
    kappa_vals = []
    for i, cls in enumerate(class_names):
        pos = int(y_true[:, i].sum())
        if 0 < pos < len(y_true):
            k = float(cohen_kappa_score(
                y_true[:, i], y_pred_binary[:, i], weights="quadratic"
            ))
            per_class_kappa[cls] = k
            kappa_vals.append(k)
        else:
            per_class_kappa[cls] = None
    kappa = float(np.mean(kappa_vals)) if kappa_vals else 0.0

    # ---- sensitivity / specificity (per class) ----
    sensitivity = {}
    specificity = {}
    for i, cls in enumerate(class_names):
        tp = int(((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fn = int(((y_pred_binary[:, i] == 0) & (y_true[:, i] == 1)).sum())
        tn = int(((y_pred_binary[:, i] == 0) & (y_true[:, i] == 0)).sum())
        fp = int(((y_pred_binary[:, i] == 1) & (y_true[:, i] == 0)).sum())
        sensitivity[cls] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity[cls] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "kappa": kappa,
        "per_class_kappa": per_class_kappa,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: dict,
    save_dir: str,
    experiment_name: str,
    config_snapshot: Optional[dict] = None,
) -> str:
    """Serialise metrics dict to JSON.

    Args:
        metrics         : output of compute_metrics()
        save_dir        : directory to write into
        experiment_name : used as filename stem
        config_snapshot : optional config dict to embed

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Deep-copy + convert all numpy scalars to native Python types
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    payload = _convert(metrics)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    if config_snapshot is not None:
        payload["config_snapshot"] = config_snapshot

    out_path = os.path.join(save_dir, f"{experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# evaluate_with_tta
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_tta(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    thresholds: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
) -> dict:
    """Evaluate model with horizontal-flip TTA.

    For each batch: forward original + hflip image, average sigmoid probs.
    If thresholds is None, calls find_optimal_thresholds on the collected probs.

    Args:
        model       : trained nn.Module (eval mode)
        loader      : DataLoader yielding (images, labels)
        device      : torch.device
        num_classes : number of output classes
        thresholds  : pre-computed thresholds or None
        class_names : list of class name strings; defaults to ["0", "1", ...]

    Returns:
        dict with keys: probs, labels, thresholds, and all metric keys
                        from compute_metrics() (macro_auc, per_class_auc,
                        macro_f1, per_class_f1, kappa, per_class_kappa,
                        sensitivity, specificity).
    """
    model.eval()
    all_probs = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        # Original forward
        logits_orig = model(imgs)
        # Horizontal-flip forward
        logits_flip = model(torch.flip(imgs, dims=[3]))

        probs_batch = (
            torch.sigmoid(logits_orig) + torch.sigmoid(logits_flip)
        ) / 2.0

        all_probs.append(probs_batch.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)

    probs = np.concatenate(all_probs, axis=0)    # (N, C)
    labels = np.concatenate(all_labels, axis=0)  # (N, C)

    if thresholds is None:
        thresholds = find_optimal_thresholds(probs, labels, num_classes)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    metrics = compute_metrics(labels, probs, thresholds, class_names)

    return {
        "probs": probs,
        "labels": labels,
        "thresholds": thresholds,
        **metrics,
    }

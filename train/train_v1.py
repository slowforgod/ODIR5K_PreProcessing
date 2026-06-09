"""
train/train_v1.py

V1 앙상블 추론 전용 스크립트.
V0 / V1_L / V1_LA / V1_LB / V1_LAB 5개 variant의 best.pth를 로드하여
soft routing (uniform average) + hard routing 앙상블 후 결과 JSON 저장.

훈련은 각각 train_v0 / train_v1_L / train_v1_LA / train_v1_LB / train_v1_LAB 로 진행.

Usage:
    python -m train.train_v1 --config configs/v1.yaml
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import yaml

warnings.filterwarnings("ignore")

from preprocessing.common import crop_black_border

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M"]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# variant별 preprocessor 매핑
VARIANT_PREPROCESSOR = {
    "V0":     ("preprocessing.v0.preprocess",   "V0Preprocessor",  {}),
    "V1_L":   ("preprocessing.v1.clahe_L",      "V1LPreprocessor", {}),
    "V1_LA":  ("preprocessing.v1.clahe_LA",     "V1LAPreprocessor", {}),
    "V1_LB":  ("preprocessing.v1.clahe_LB",     "V1LBPreprocessor", {}),
    "V1_LAB": ("preprocessing.v1.clahe_LAB",    "V1LABPreprocessor", {}),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _build_preprocessor(variant: str, clip_limit: float, tile_grid: tuple):
    """variant 이름에 맞는 preprocessor 인스턴스 반환."""
    import importlib
    module_path, cls_name, _ = VARIANT_PREPROCESSOR[variant]
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    # V0는 kwargs 없음, V1_*는 clip_limit / tile_grid_size
    if variant == "V0":
        return cls()
    return cls(clip_limit=clip_limit, tile_grid_size=tile_grid)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ODIRV1Dataset(Dataset):
    def __init__(
        self,
        df,
        image_dir: str,
        class_cols: List[str],
        img_size: int = 224,
        preprocessor=None,
    ):
        self.preprocessor = preprocessor
        self.img_size = img_size
        self._to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, str(row["filename"]))
            if not os.path.isfile(img_path):
                missing += 1
                continue
            label = row[class_cols].values.astype(np.float32)
            self.samples.append((img_path, label))

        if missing:
            print(f"  [warn] {missing} rows skipped (image file not found)", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img_np = crop_black_border(img_np)

        if self.preprocessor is not None:
            img_np, _ = self.preprocessor.apply(img_np, label)

        img_pil = Image.fromarray(img_np).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        return self._to_tensor_norm(img_pil), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# TTA 추론
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_tta(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        p_orig = torch.sigmoid(model(imgs))
        p_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
        all_probs.append(((p_orig + p_flip) * 0.5).cpu().numpy())
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v1.routing import (
        auto_route_from_val,
        hard_route,
        soft_route_uniform,
    )
    from analysis.metrics import compute_metrics, find_optimal_thresholds

    # ── device ──────────────────────────────────────────────────────────
    requested = cfg["train"].get("device", "cpu")
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as fh:
        split_json = json.load(fh)

    val_ids  = set(split_json[data_cfg["split_val_key"]])
    test_key = data_cfg.get("split_test_key")
    test_ids = set(split_json[test_key]) if test_key and test_key in split_json else None

    val_df  = df[df[data_cfg["patient_id_col"]].isin(val_ids)].reset_index(drop=True)
    test_df = (
        df[df[data_cfg["patient_id_col"]].isin(test_ids)].reset_index(drop=True)
        if test_ids is not None else None
    )

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 224))
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = cfg.get("num_classes", 7)
    batch_size  = int(cfg["train"].get("batch_size", 32))
    num_workers = int(cfg["train"].get("num_workers", 4))
    dropout     = float(cfg["train"].get("dropout", 0.5))

    pre_cfg    = cfg.get("preprocessing", {})
    clip_limit = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid  = tuple(pre_cfg.get("tile_grid_size", [8, 8]))

    variant_cfgs = cfg.get("variants", {})

    print(f"Val  : {len(val_df)} samples")
    if test_df is not None:
        print(f"Test : {len(test_df)} samples")

    probs_val_by_variant:  Dict[str, np.ndarray] = {}
    probs_test_by_variant: Dict[str, np.ndarray] = {}
    val_labels_shared:     Optional[np.ndarray] = None
    test_labels_shared:    Optional[np.ndarray] = None
    per_variant_val:       Dict[str, dict] = {}
    per_variant_test:      Dict[str, dict] = {}
    thresholds_by_variant: Dict[str, np.ndarray] = {}

    # ── 각 variant 추론 ──────────────────────────────────────────────────
    for variant, variant_dir in variant_cfgs.items():
        print(f"\n{'='*60}")
        print(f"[{variant}] Loading from {variant_dir}")

        ckpt_path = os.path.join(variant_dir, "best.pth")
        thr_path  = os.path.join(variant_dir, "thresholds.npy")

        if not os.path.isfile(ckpt_path):
            print(f"  [skip] best.pth not found: {ckpt_path}")
            continue
        if not os.path.isfile(thr_path):
            print(f"  [skip] thresholds.npy not found: {thr_path}")
            continue

        model = build_model(
            name=cfg["model_name"],
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        thresholds = np.load(thr_path)
        thresholds_by_variant[variant] = thresholds

        preprocessor = _build_preprocessor(variant, clip_limit, tile_grid)

        # val 추론
        val_ds = ODIRV1Dataset(val_df, image_dir, class_cols, img_size, preprocessor)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_probs, val_labels = _infer_tta(model, val_loader, device)

        if val_labels_shared is None:
            val_labels_shared = val_labels
        probs_val_by_variant[variant] = val_probs

        val_metrics = compute_metrics(val_labels, val_probs, thresholds, class_names)
        per_variant_val[variant] = {**val_metrics, "thresholds": thresholds.tolist()}
        print(f"  [val]  auc={val_metrics['macro_auc']:.4f}  f1={val_metrics['macro_f1']:.4f}  kappa={val_metrics['kappa']:.4f}")

        # test 추론
        if test_df is not None:
            test_ds = ODIRV1Dataset(test_df, image_dir, class_cols, img_size, preprocessor)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_probs, test_labels = _infer_tta(model, test_loader, device)

            if test_labels_shared is None:
                test_labels_shared = test_labels
            probs_test_by_variant[variant] = test_probs

            test_metrics = compute_metrics(test_labels, test_probs, thresholds, class_names)
            per_variant_test[variant] = test_metrics
            print(f"  [test] auc={test_metrics['macro_auc']:.4f}  f1={test_metrics['macro_f1']:.4f}  kappa={test_metrics['kappa']:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert val_labels_shared is not None, "No variant produced a valid result."

    # ── soft routing 앙상블 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Soft routing (uniform average)...")

    prob_soft_val = soft_route_uniform(probs_val_by_variant, class_names)
    thr_soft      = find_optimal_thresholds(prob_soft_val, val_labels_shared, num_classes)
    met_soft_val  = compute_metrics(val_labels_shared, prob_soft_val, thr_soft, class_names)
    print(f"  [val]  auc={met_soft_val['macro_auc']:.4f}  f1={met_soft_val['macro_f1']:.4f}  kappa={met_soft_val['kappa']:.4f}")

    met_soft_test = None
    if test_labels_shared is not None:
        prob_soft_test = soft_route_uniform(probs_test_by_variant, class_names)
        met_soft_test  = compute_metrics(test_labels_shared, prob_soft_test, thr_soft, class_names)
        print(f"  [test] auc={met_soft_test['macro_auc']:.4f}  f1={met_soft_test['macro_f1']:.4f}  kappa={met_soft_test['kappa']:.4f}")

    # ── auto hard routing (val F1 기준으로 클래스별 best variant 자동 선정) ──
    print(f"\n{'='*60}")
    print("Auto hard routing (per-class best variant from val F1)...")

    routing_hard = auto_route_from_val(
        probs_val_by_variant,
        val_labels_shared,
        class_names,
        thresholds_by_variant=thresholds_by_variant,
    )
    for cls in class_names:
        print(f"  {cls}: {routing_hard[cls]}")

    prob_hard_val = hard_route(probs_val_by_variant, class_names, routing_hard)
    thr_hard      = find_optimal_thresholds(prob_hard_val, val_labels_shared, num_classes)
    met_hard_val  = compute_metrics(val_labels_shared, prob_hard_val, thr_hard, class_names)
    print(f"  [val]  auc={met_hard_val['macro_auc']:.4f}  f1={met_hard_val['macro_f1']:.4f}  kappa={met_hard_val['kappa']:.4f}")

    met_hard_test = None
    if test_labels_shared is not None:
        prob_hard_test = hard_route(probs_test_by_variant, class_names, routing_hard)
        met_hard_test  = compute_metrics(test_labels_shared, prob_hard_test, thr_hard, class_names)
        print(f"  [test] auc={met_hard_test['macro_auc']:.4f}  f1={met_hard_test['macro_f1']:.4f}  kappa={met_hard_test['kappa']:.4f}")

    # ── 결과 저장 ────────────────────────────────────────────────────────
    output_base = cfg.get("output_dir", "analysis/v1_analysis")
    exp_name    = cfg.get("experiment_name", "v1_resnet50_seed42")
    os.makedirs(output_base, exist_ok=True)

    combined = {
        "experiment_name": exp_name,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "config_snapshot": cfg,
        "split":           "8:1:1",
        "val": {
            "per_variant_metrics": _to_serializable(per_variant_val),
            "soft_routing": _to_serializable({**met_soft_val, "thresholds": thr_soft.tolist()}),
            "hard_routing": _to_serializable({**met_hard_val, "routing_map": routing_hard, "thresholds": thr_hard.tolist()}),
        },
        "test": {
            "per_variant_metrics": _to_serializable(per_variant_test),
            "soft_routing": _to_serializable(met_soft_test),
            "hard_routing": _to_serializable({**met_hard_test, "routing_map": routing_hard} if met_hard_test else None),
        } if test_labels_shared is not None else None,
    }

    json_path = os.path.join(output_base, f"{exp_name}.json")
    with open(json_path, "w") as fh:
        json.dump(combined, fh, indent=2)

    print(f"\nResults saved → {json_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V1 ensemble inference (soft + hard routing)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
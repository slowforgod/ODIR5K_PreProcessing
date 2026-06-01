"""
train/train_v1.py

Entry point for the redesigned V1 experiment: 5 CLAHE variants + per-class routing.

Trains one ResNet-50 per variant sequentially, then combines results via
hard routing and uniform soft routing on the validation set.
Test set is evaluated with the best thresholds from val (no recomputation).

Usage:
    python -m train.train_v1 --config configs/v1.yaml
    python -m train.train_v1 --config configs/v1.yaml --smoke

변경 이력:
  - split JSON: patient_split_7class_stratified.json (8:2)
              → patient_split_7class_stratified_811.json (8:1:1)
  - test split 로드 + test_loader 구성
  - 각 variant 학습 후 test 평가 추가 (val threshold 재사용)
  - 최종 JSON에 per_variant_test_metrics, routing test 결과 포함
"""

import argparse
import csv
import json
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import yaml

warnings.filterwarnings("ignore")

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M"]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        smoke_n: int = 0,
    ):
        if smoke_n > 0:
            df = df.head(smoke_n)

        self.preprocessor = preprocessor
        self._resize = T.Resize((img_size, img_size))
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
        img_pil = self._resize(Image.open(img_path).convert("RGB"))

        if self.preprocessor is not None:
            img_np = np.array(img_pil, dtype=np.uint8)
            img_np, _ = self.preprocessor.apply(img_np, label)
            img_tensor = self._to_tensor_norm(img_np)
        else:
            img_tensor = self._to_tensor_norm(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def _compute_label_stats(dataset: ODIRV1Dataset) -> tuple:
    labels = np.stack([s[1] for s in dataset.samples])
    class_freq = labels.sum(axis=0).clip(min=1.0)
    class_inv = 1.0 / class_freq
    sample_weights = torch.tensor(
        (labels * class_inv).sum(axis=1), dtype=torch.float32
    )
    pos_weight = torch.tensor(
        (len(labels) - class_freq) / class_freq, dtype=torch.float32
    )
    return sample_weights, pos_weight


# ---------------------------------------------------------------------------
# Training loop (per variant)
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _validate_tta(model, loader, device):
    """Horizontal-flip TTA. Returns (probs, labels) ndarrays."""
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

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v1.clahe import V1Preprocessor
    from preprocessing.v1.routing import (
        DEFAULT_HARD_ROUTING,
        DEFAULT_VARIANTS,
        hard_route,
        soft_route_uniform,
    )
    from analysis.metrics import (
        compute_metrics,
        evaluate_with_tta,
        find_optimal_thresholds,
    )

    # ------------------------------------------------------------------ #
    # Seed + device                                                        #
    # ------------------------------------------------------------------ #
    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

    requested = cfg["train"].get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA requested but not available — using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(requested if torch.cuda.is_available() else "cpu")

    use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"

    # ------------------------------------------------------------------ #
    # Data loading + split  ★ test 추가                                   #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as fh:
        split_json = json.load(fh)

    train_ids = set(split_json[data_cfg["split_train_key"]])
    val_ids   = set(split_json[data_cfg["split_val_key"]])
    # ★ test_patients — config에 키가 있을 때만 로드 (하위 호환)
    test_key  = data_cfg.get("split_test_key")
    test_ids  = set(split_json[test_key]) if test_key and test_key in split_json else None

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[data_cfg["patient_id_col"]].isin(val_ids)].reset_index(drop=True)
    test_df  = (
        df[df[data_cfg["patient_id_col"]].isin(test_ids)].reset_index(drop=True)
        if test_ids is not None else None
    )

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 224))
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = cfg.get("num_classes", 7)

    smoke_train = 200 if smoke else 0
    smoke_val   = 50  if smoke else 0
    smoke_test  = 50  if smoke else 0

    pre_cfg    = cfg.get("preprocessing", {})
    variants   = pre_cfg.get("variants", DEFAULT_VARIANTS)
    clip_limit = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid  = tuple(pre_cfg.get("tile_grid_size", [8, 8]))

    train_cfg    = cfg["train"]
    batch_size   = int(train_cfg.get("batch_size", 32))
    num_epochs   = 2 if smoke else int(train_cfg.get("num_epochs", 20))
    patience     = int(train_cfg.get("patience", 8))
    lr           = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    dropout      = float(train_cfg.get("dropout", 0.5))
    num_workers  = 0 if smoke else int(train_cfg.get("num_workers", 4))

    output_base = cfg.get("output_dir", "analysis/v1_analysis")
    exp_name    = cfg.get("experiment_name", "v1_resnet50_seed42")
    os.makedirs(output_base, exist_ok=True)

    routing_cfg = cfg.get("routing", {}).get("hard", DEFAULT_HARD_ROUTING)
    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    per_variant_metrics:      Dict[str, dict] = {}
    per_variant_test_metrics: Dict[str, dict] = {}   # ★ test
    probs_by_variant:         Dict[str, np.ndarray] = {}
    test_probs_by_variant:    Dict[str, np.ndarray] = {}  # ★ test
    val_labels_shared:        Optional[np.ndarray] = None
    test_labels_shared:       Optional[np.ndarray] = None  # ★ test

    cached_sampler_weights: Optional[torch.Tensor] = None
    cached_pos_weight_dev:  Optional[torch.Tensor] = None

    print(f"  Train : {len(train_df)} samples")
    print(f"  Val   : {len(val_df)} samples")
    if test_df is not None:
        print(f"  Test  : {len(test_df)} samples")

    # ------------------------------------------------------------------ #
    # 5-variant training loop                                              #
    # ------------------------------------------------------------------ #
    for v_idx, variant in enumerate(variants, start=1):
        print(f"\n{'='*60}")
        print(f"[Variant {v_idx}/{len(variants)}: {variant}]")
        print(f"{'='*60}")

        variant_dir = os.path.join(output_base, variant)
        os.makedirs(variant_dir, exist_ok=True)

        preprocessor = V1Preprocessor(
            variant=variant,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid,
        )

        train_ds = ODIRV1Dataset(
            train_df, image_dir, class_cols,
            img_size=img_size, preprocessor=preprocessor, smoke_n=smoke_train,
        )
        val_ds = ODIRV1Dataset(
            val_df, image_dir, class_cols,
            img_size=img_size, preprocessor=preprocessor, smoke_n=smoke_val,
        )
        # ★ test_ds
        test_ds = (
            ODIRV1Dataset(
                test_df, image_dir, class_cols,
                img_size=img_size, preprocessor=preprocessor, smoke_n=smoke_test,
            ) if test_df is not None else None
        )

        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}"
              + (f"  Test: {len(test_ds)}" if test_ds else ""))

        if cached_sampler_weights is None:
            cached_sampler_weights, pos_weight_cpu = _compute_label_stats(train_ds)
            cached_pos_weight_dev = pos_weight_cpu.to(device)

        sampler = WeightedRandomSampler(
            cached_sampler_weights,
            num_samples=len(cached_sampler_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        # ★ test_loader
        test_loader = (
            DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=(device.type == "cuda"),
            ) if test_ds is not None else None
        )

        seed_everything(seed)
        model = build_model(
            name=cfg["model_name"],
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=cached_pos_weight_dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler    = torch.amp.GradScaler("cuda") if use_amp else None

        history_path   = os.path.join(variant_dir, "history.csv")
        history_fields = ["epoch", "train_loss", "val_loss", "val_macro_auc", "val_macro_f1", "val_kappa"]
        with open(history_path, "w", newline="") as hf:
            csv.DictWriter(hf, fieldnames=history_fields).writeheader()

        best_auc   = -1.0
        no_improve = 0
        best_epoch = 0
        ckpt_path  = os.path.join(variant_dir, "best.pth")

        for epoch in range(1, num_epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_probs, val_labels = _validate_tta(model, val_loader, device)

            val_loss = float(
                nn.BCELoss()(
                    torch.tensor(val_probs, dtype=torch.float32),
                    torch.tensor(val_labels, dtype=torch.float32),
                ).item()
            )
            epoch_metrics = compute_metrics(val_labels, val_probs, log_thresholds, class_names)
            val_auc   = epoch_metrics["macro_auc"]
            val_f1    = epoch_metrics["macro_f1"]
            val_kappa = epoch_metrics["kappa"]

            scheduler.step()

            print(
                f"  Epoch {epoch:03d}/{num_epochs} | "
                f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | val_kappa={val_kappa:.4f}"
            )

            with open(history_path, "a", newline="") as hf:
                writer = csv.DictWriter(hf, fieldnames=history_fields)
                writer.writerow({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_macro_auc": round(val_auc, 6),
                    "val_macro_f1": round(val_f1, 6),
                    "val_kappa": round(val_kappa, 6),
                })

            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"    [saved] best.pth (val_auc={best_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [early stop] no improvement for {patience} epochs")
                    break

        # ---- Post-training: reload best.pth, val TTA eval ----
        if best_auc >= 0 and os.path.isfile(ckpt_path):
            print(f"\n  [reload] Loading best.pth (epoch {best_epoch}) ...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            # val 평가 + threshold 최적화
            eval_result = evaluate_with_tta(
                model, val_loader, device, num_classes,
                thresholds=None,          # val에서 threshold 최적화
                class_names=class_names,
            )
            thresholds = eval_result["thresholds"]
            np.save(os.path.join(variant_dir, "thresholds.npy"), thresholds)

            probs_by_variant[variant] = eval_result["probs"]
            if val_labels_shared is None:
                val_labels_shared = eval_result["labels"]

            per_variant_metrics[variant] = {
                "macro_auc":     eval_result["macro_auc"],
                "per_class_auc": eval_result["per_class_auc"],
                "macro_f1":      eval_result["macro_f1"],
                "kappa":         eval_result["kappa"],
                "sensitivity":   eval_result["sensitivity"],
                "specificity":   eval_result["specificity"],
                "thresholds":    thresholds.tolist(),
                "best_epoch":    best_epoch,
            }
            print(
                f"  [val]  auc={eval_result['macro_auc']:.4f}  "
                f"f1={eval_result['macro_f1']:.4f}  "
                f"kappa={eval_result['kappa']:.4f}"
            )

            # ★ test 평가 — val threshold 재사용 (재계산 금지)
            if test_loader is not None:
                test_result = evaluate_with_tta(
                    model, test_loader, device, num_classes,
                    thresholds=thresholds,   # val threshold 재사용
                    class_names=class_names,
                )
                test_probs_by_variant[variant] = test_result["probs"]
                if test_labels_shared is None:
                    test_labels_shared = test_result["labels"]

                per_variant_test_metrics[variant] = {
                    "macro_auc":     test_result["macro_auc"],
                    "per_class_auc": test_result["per_class_auc"],
                    "macro_f1":      test_result["macro_f1"],
                    "kappa":         test_result["kappa"],
                    "sensitivity":   test_result["sensitivity"],
                    "specificity":   test_result["specificity"],
                }
                print(
                    f"  [test] auc={test_result['macro_auc']:.4f}  "
                    f"f1={test_result['macro_f1']:.4f}  "
                    f"kappa={test_result['kappa']:.4f}"
                )
        else:
            print(f"  [warn] No improvement recorded for variant {variant}.")
            probs_by_variant[variant] = np.zeros(
                (len(val_ds), num_classes), dtype=np.float32
            )
            per_variant_metrics[variant] = {
                "macro_auc": None, "per_class_auc": {},
                "macro_f1": None, "kappa": None,
                "sensitivity": {}, "specificity": {},
                "thresholds": [], "best_epoch": 0,
            }
            if test_loader is not None:
                test_probs_by_variant[variant] = np.zeros(
                    (len(test_ds), num_classes), dtype=np.float32
                )
                per_variant_test_metrics[variant] = {
                    "macro_auc": None, "per_class_auc": {},
                    "macro_f1": None, "kappa": None,
                    "sensitivity": {}, "specificity": {},
                }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Routing aggregation — val                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Computing routing metrics...")
    assert val_labels_shared is not None, "No variant produced a valid eval result."

    routing_metrics: Dict[str, dict] = {}

    prob_hard = hard_route(probs_by_variant, class_names, routing_cfg)
    thr_hard  = find_optimal_thresholds(prob_hard, val_labels_shared, num_classes)
    met_hard  = compute_metrics(val_labels_shared, prob_hard, thr_hard, class_names)
    routing_metrics["hard"] = {
        "routing_map":   routing_cfg,
        "macro_auc":     met_hard["macro_auc"],
        "per_class_auc": met_hard["per_class_auc"],
        "macro_f1":      met_hard["macro_f1"],
        "kappa":         met_hard["kappa"],
        "sensitivity":   met_hard["sensitivity"],
        "specificity":   met_hard["specificity"],
        "thresholds":    thr_hard.tolist(),
    }

    prob_soft = soft_route_uniform(probs_by_variant, class_names)
    thr_soft  = find_optimal_thresholds(prob_soft, val_labels_shared, num_classes)
    met_soft  = compute_metrics(val_labels_shared, prob_soft, thr_soft, class_names)
    routing_metrics["soft_uniform"] = {
        "weighting":     "uniform across all 5 variants",
        "macro_auc":     met_soft["macro_auc"],
        "per_class_auc": met_soft["per_class_auc"],
        "macro_f1":      met_soft["macro_f1"],
        "kappa":         met_soft["kappa"],
        "sensitivity":   met_soft["sensitivity"],
        "specificity":   met_soft["specificity"],
        "thresholds":    thr_soft.tolist(),
    }

    print(f"  [val] Hard routing   → auc={met_hard['macro_auc']:.4f}  f1={met_hard['macro_f1']:.4f}")
    print(f"  [val] Soft (uniform) → auc={met_soft['macro_auc']:.4f}  f1={met_soft['macro_f1']:.4f}")

    # ★ Routing aggregation — test
    test_routing_metrics: Dict[str, dict] = {}
    if test_labels_shared is not None:
        # hard routing test: val threshold 재사용
        prob_hard_test = hard_route(test_probs_by_variant, class_names, routing_cfg)
        met_hard_test  = compute_metrics(test_labels_shared, prob_hard_test, thr_hard, class_names)
        test_routing_metrics["hard"] = {
            "routing_map":   routing_cfg,
            "macro_auc":     met_hard_test["macro_auc"],
            "per_class_auc": met_hard_test["per_class_auc"],
            "macro_f1":      met_hard_test["macro_f1"],
            "kappa":         met_hard_test["kappa"],
            "sensitivity":   met_hard_test["sensitivity"],
            "specificity":   met_hard_test["specificity"],
        }

        # soft routing test: val threshold 재사용
        prob_soft_test = soft_route_uniform(test_probs_by_variant, class_names)
        met_soft_test  = compute_metrics(test_labels_shared, prob_soft_test, thr_soft, class_names)
        test_routing_metrics["soft_uniform"] = {
            "weighting":     "uniform across all 5 variants",
            "macro_auc":     met_soft_test["macro_auc"],
            "per_class_auc": met_soft_test["per_class_auc"],
            "macro_f1":      met_soft_test["macro_f1"],
            "kappa":         met_soft_test["kappa"],
            "sensitivity":   met_soft_test["sensitivity"],
            "specificity":   met_soft_test["specificity"],
        }

        print(f"  [test] Hard routing   → auc={met_hard_test['macro_auc']:.4f}  f1={met_hard_test['macro_f1']:.4f}")
        print(f"  [test] Soft (uniform) → auc={met_soft_test['macro_auc']:.4f}  f1={met_soft_test['macro_f1']:.4f}")

    # ------------------------------------------------------------------ #
    # Save combined JSON                                                   #
    # ------------------------------------------------------------------ #
    combined = {
        "experiment_name":         exp_name,
        "timestamp":               datetime.utcnow().isoformat() + "Z",
        "config_snapshot":         cfg,
        "split":                   "8:1:1",
        "val": {
            "per_variant_metrics": _to_serializable(per_variant_metrics),
            "routing_metrics":     _to_serializable(routing_metrics),
        },
        "test": {
            "per_variant_metrics": _to_serializable(per_variant_test_metrics),
            "routing_metrics":     _to_serializable(test_routing_metrics),
        } if test_labels_shared is not None else None,
    }

    json_path = os.path.join(output_base, f"{exp_name}.json")
    with open(json_path, "w") as fh:
        json.dump(combined, fh, indent=2)

    print(f"\nCombined results → {json_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V1 multi-variant CLAHE experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 200 train / 50 val, 2 epochs per variant",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
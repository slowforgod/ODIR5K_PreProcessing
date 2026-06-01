"""
train/train_v6.py

Entry point for V6: V1 × V2 (LSE + Aug) × Manifold Mixup × 5 losses.

The "full stack" combination:
  - V1 CLAHE variants (dataset-side, 5 variants)
  - V2 LSE oversampling + albumentations augmentation (training only)
  - V3 Manifold Mixup (trainer-level, applied in train_one_epoch)
  - 5 loss functions compared per variant

Val set: V1 CLAHE only (no V2 augmentation — guardrail #4).
Optimizer: Adam (V2 recipe — V6 uses V2 training protocol).
Sampler: None / shuffle=True (LSE replaces WeightedRandomSampler).
CB weights: from original train_df BEFORE LSE expansion (guardrail #3).

Outer loop: 5 CLAHE variants
Inner loop: 5 loss functions
Total: 25 training runs.

Usage:
    python -m train.train_v6 --config configs/v6.yaml
    python -m train.train_v6 --config configs/v6.yaml --smoke
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
from torch.utils.data import DataLoader, Dataset
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
# Datasets
# ---------------------------------------------------------------------------

class ODIRV6TrainDataset(Dataset):
    """Training dataset for V6: LSE records → V6Preprocessor (V1+V2) → tensor."""

    def __init__(
        self,
        records: List[dict],
        image_dir: str,
        img_size: int = 224,
        preprocessor=None,
        smoke_n: int = 0,
    ):
        if smoke_n > 0:
            records = records[:smoke_n]

        self.preprocessor = preprocessor

        # Resize BEFORE CLAHE so V1 and V6 train both operate on 224×224
        self._resize = T.Resize((img_size, img_size))
        self._to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.samples = []
        missing = 0
        for rec in records:
            img_path = os.path.join(image_dir, rec["filename"])
            if not os.path.isfile(img_path):
                missing += 1
                continue
            self.samples.append((img_path, rec["labels"].copy()))

        if missing:
            print(
                f"  [warn] {missing} records skipped (image file not found)",
                file=sys.stderr,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_pil = self._resize(Image.open(img_path).convert("RGB"))  # 512 → 224 먼저
        img_np  = np.array(img_pil, dtype=np.uint8)

        if self.preprocessor is not None:
            img_np, _ = self.preprocessor.apply(img_np, label)  # CLAHE on 224×224

        img_tensor = self._to_tensor_norm(img_np)
        return img_tensor, torch.tensor(label, dtype=torch.float32)


class ODIRV6ValDataset(Dataset):
    """Validation dataset for V6: V1 CLAHE only (no V2 augmentation)."""

    def __init__(
        self,
        records: List[dict],
        image_dir: str,
        img_size: int = 224,
        preprocessor=None,
        smoke_n: int = 0,
    ):
        if smoke_n > 0:
            records = records[:smoke_n]

        self.preprocessor = preprocessor

        self._resize = T.Resize((img_size, img_size))
        self._to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.samples = []
        missing = 0
        for rec in records:
            img_path = os.path.join(image_dir, rec["filename"])
            if not os.path.isfile(img_path):
                missing += 1
                continue
            self.samples.append((img_path, rec["labels"].copy()))

        if missing:
            print(
                f"  [warn] {missing} records skipped (image file not found)",
                file=sys.stderr,
            )

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
# Training loop — Manifold Mixup (V3 pattern) with Adam (V2 rule)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    alpha: float,
    mixup_layers: list,
    scaler=None,
    mixup_prob: float = 1.0,
    grad_clip: float = 0.0,
):
    """One training epoch with Manifold Mixup for V6.

    Args:
        mixup_prob : probability of applying Manifold Mixup per batch.
                     Remaining batches do a plain forward to keep raw label
                     distribution in the gradient signal.
        grad_clip  : max gradient norm (0 disables). AMP-aware.
    """
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        do_mixup = (alpha > 0.0) and (np.random.rand() < mixup_prob)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                if do_mixup:
                    logits, label_a, label_b, lam = model(
                        imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers,
                    )
                    # NaN guard: ensure λ ≥ 0.5 so dominant term never vanishes
                    lam = max(lam, 1.0 - lam)
                    loss = lam * criterion(logits, label_a) + (1.0 - lam) * criterion(logits, label_b)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if do_mixup:
                logits, label_a, label_b, lam = model(
                    imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers,
                )
                lam = max(lam, 1.0 - lam)
                loss = lam * criterion(logits, label_a) + (1.0 - lam) * criterion(logits, label_b)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v1.clahe import V1Preprocessor
    from preprocessing.v1.routing import DEFAULT_VARIANTS, soft_route_uniform
    from preprocessing.v2.augmentation import V2Preprocessor
    from preprocessing.v2.oversampling import expand_with_lse
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
    from preprocessing.v6.preprocess import V6Preprocessor
    from analysis.metrics import (
        compute_metrics,
        evaluate_with_tta,
        find_optimal_thresholds,
    )

    # ------------------------------------------------------------------ #
    # Seed + device + AMP                                                 #
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
    # Data loading + split                                                #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as fh:
        split_json = json.load(fh)

    train_ids = set(split_json[data_cfg["split_train_key"]])
    val_ids   = set(split_json[data_cfg["split_val_key"]])

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[data_cfg["patient_id_col"]].isin(val_ids)].reset_index(drop=True)

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 224))
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = cfg.get("num_classes", 7)

    # ------------------------------------------------------------------ #
    # CRITICAL: CB weights from original train_df (pre-LSE)              #
    # ------------------------------------------------------------------ #
    labels_ref_np = train_df[class_cols].values.astype(np.float32)
    cb_beta       = float(cfg.get("losses", {}).get("cb_beta", 0.9999))
    cb_normalize  = cfg.get("losses", {}).get("cb_weight_normalize", "mean_one")
    cb_weights    = effective_number_class_weights(
        labels_ref_np, beta=cb_beta, normalize=cb_normalize
    )
    print(f"[CB weights from original train_df ({len(train_df)} rows)]")
    for i, c in enumerate(class_cols):
        print(f"  {c}: {cb_weights[i]:.4f}")

    # ------------------------------------------------------------------ #
    # LSE expansion — done ONCE (records reused for all 25 runs)         #
    # ------------------------------------------------------------------ #
    pre_cfg      = cfg.get("preprocessing", {})
    target_count = int(pre_cfg.get("lse_custom_target_count", 850))
    max_aug      = int(pre_cfg.get("lse_max_aug_per_source", 12))

    expanded_records = expand_with_lse(
        train_df,
        class_cols=class_cols,
        target_count=target_count,
        max_aug_per_source=max_aug,
        seed=seed,
    )

    # Val records: no LSE, val_df rows only
    val_records = [
        {
            "filename": str(row["filename"]),
            "labels": row[class_cols].values.astype(np.float32),
            "aug_idx": 0,
        }
        for _, row in val_df.iterrows()
    ]

    # ------------------------------------------------------------------ #
    # Config extraction                                                   #
    # ------------------------------------------------------------------ #
    smoke_train = 200 if smoke else 0
    smoke_val   = 50  if smoke else 0

    variants    = pre_cfg.get("variants", DEFAULT_VARIANTS)
    clip_limit  = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid   = tuple(pre_cfg.get("tile_grid_size", [8, 8]))
    alpha        = float(pre_cfg.get("mixup_alpha", 0.2))
    mixup_layers = list(pre_cfg.get("mixup_layers", [0, 1, 2, 3]))
    mixup_prob   = float(pre_cfg.get("mixup_prob", 1.0))

    aug_kwargs = {
        k: pre_cfg[k]
        for k in (
            "hflip_p", "vflip_p", "geo_p",
            "rotate_limit", "affine_scale", "affine_translate",
            "affine_rotate", "affine_shear", "crop_scale",
            "color_jitter_p", "brightness", "contrast", "saturation",
            "blur_p", "blur_limit",
        )
        if k in pre_cfg
    }

    train_cfg    = cfg["train"]
    batch_size   = int(train_cfg.get("batch_size", 32))
    num_epochs   = 2 if smoke else int(train_cfg.get("num_epochs", 20))
    patience     = int(train_cfg.get("patience", 8))
    lr           = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    dropout      = float(train_cfg.get("dropout", 0.5))
    num_workers  = 0 if smoke else int(train_cfg.get("num_workers", 4))
    grad_clip    = float(train_cfg.get("grad_clip", 0.0))
    eta_min      = float(train_cfg.get("eta_min", 0.0))

    loss_experiments = cfg.get("losses", {}).get(
        "experiments",
        ["bce_with_logits_loss", "focal_loss", "robust_asymmetric_loss",
         "cb_bce_with_logits_loss", "cb_focal_loss"],
    )
    loss_params = {k: cfg.get("losses", {}).get(k) for k in cfg.get("losses", {})}

    output_base = cfg.get("output_dir", "analysis/v6_analysis")
    exp_name    = cfg.get("experiment_name", "v6_resnet50_seed42")
    os.makedirs(output_base, exist_ok=True)

    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    # Accumulators
    probs_by_loss_by_variant: Dict[str, Dict[str, np.ndarray]] = {
        ln: {} for ln in loss_experiments
    }
    per_variant_per_loss_metrics: Dict[str, Dict[str, dict]] = {}
    val_labels_shared: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Outer loop: 5 variants                                              #
    # ------------------------------------------------------------------ #
    for v_idx, variant in enumerate(variants, start=1):
        print(f"\n{'='*60}")
        print(f"[Variant {v_idx}/{len(variants)}: {variant}]")
        print(f"{'='*60}")

        per_variant_per_loss_metrics[variant] = {}

        # ---- Preprocessors ----
        v6_pre = V6Preprocessor(
            v1_kwargs=dict(variant=variant, clip_limit=clip_limit, tile_grid_size=tile_grid),
            v2_kwargs=dict(img_size=img_size, **aug_kwargs),
        )
        # Validation: V1 only (guardrail #4)
        v1_val_pre = V1Preprocessor(
            variant=variant,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid,
        )

        # ---- Datasets ----
        train_ds = ODIRV6TrainDataset(
            expanded_records,
            image_dir=image_dir,
            img_size=img_size,
            preprocessor=v6_pre,
            smoke_n=smoke_train,
        )
        val_ds = ODIRV6ValDataset(
            val_records,
            image_dir=image_dir,
            img_size=img_size,
            preprocessor=v1_val_pre,    # V1 only — no V2 on val
            smoke_n=smoke_val,
        )

        print(f"  Train (expanded): {len(train_ds)}  Val: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,       # LSE replaces WeightedRandomSampler
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # ---------------------------------------------------------------- #
        # Inner loop: 5 losses                                             #
        # ---------------------------------------------------------------- #
        for loss_idx, loss_name in enumerate(loss_experiments, start=1):
            print(f"\n  [{variant}] Loss {loss_idx}/{len(loss_experiments)}: {loss_name}")

            loss_dir = os.path.join(output_base, variant, loss_name)
            os.makedirs(loss_dir, exist_ok=True)

            seed_everything(seed)
            model = build_model(
                name=cfg["model_name"],
                num_classes=num_classes,
                pretrained=True,
                dropout=dropout,
            ).to(device)

            criterion = make_criterion(
                loss_name,
                cb_weights=cb_weights,
                params=loss_params,
            ).to(device)

            # Adam — V6 follows V2 training recipe (not AdamW)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=eta_min,
            )
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            history_path   = os.path.join(loss_dir, "history.csv")
            history_fields = ["epoch", "train_loss", "val_loss",
                              "val_macro_auc", "val_macro_f1", "val_kappa"]
            with open(history_path, "w", newline="") as hf:
                csv.DictWriter(hf, fieldnames=history_fields).writeheader()

            best_auc   = -1.0
            no_improve = 0
            best_epoch = 0
            ckpt_path  = os.path.join(loss_dir, "best.pth")

            for epoch in range(1, num_epochs + 1):
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion,
                    device, alpha, mixup_layers, scaler,
                    mixup_prob=mixup_prob, grad_clip=grad_clip,
                )

                # Validation: standard forward (no Mixup) + HF TTA
                model.eval()
                all_probs_list  = []
                all_labels_list = []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(device, non_blocking=True)
                        probs_orig = torch.sigmoid(model(imgs))
                        probs_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
                        all_probs_list.append(((probs_orig + probs_flip) / 2.0).cpu().numpy())
                        all_labels_list.append(labels.numpy())

                val_probs  = np.concatenate(all_probs_list, axis=0)
                val_labels = np.concatenate(all_labels_list, axis=0)

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
                val_sens  = float(np.mean(list(epoch_metrics["sensitivity"].values())))
                val_spec  = float(np.mean(list(epoch_metrics["specificity"].values())))

                scheduler.step()

                print(
                    f"    Epoch {epoch:03d}/{num_epochs} | "
                    f"loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_auc={val_auc:.4f} | val_f1={val_f1:.4f} | val_kappa={val_kappa:.4f} | "
                    f"val_sens={val_sens:.4f} | val_spec={val_spec:.4f}"
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
                    print(f"      [saved] best.pth (val_auc={best_auc:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"    [early stop] no improvement for {patience} epochs")
                        break

            # Post-training: reload best, TTA, thresholds
            if best_auc >= 0 and os.path.isfile(ckpt_path):
                print(f"\n    [reload] Loading best.pth (epoch {best_epoch}) and re-running TTA...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

                eval_result = evaluate_with_tta(
                    model, val_loader, device, num_classes,
                    thresholds=None,
                    class_names=class_names,
                )
                thresholds = eval_result["thresholds"]
                np.save(os.path.join(loss_dir, "thresholds.npy"), thresholds)

                probs_by_loss_by_variant[loss_name][variant] = eval_result["probs"]
                if val_labels_shared is None:
                    val_labels_shared = eval_result["labels"]

                per_variant_per_loss_metrics[variant][loss_name] = {
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
                    f"    macro_auc={eval_result['macro_auc']:.4f}  "
                    f"macro_f1={eval_result['macro_f1']:.4f}  "
                    f"kappa={eval_result['kappa']:.4f}"
                )
            else:
                print(f"    [warn] No improvement for {variant}/{loss_name}.")
                n_val = len(val_ds)
                probs_by_loss_by_variant[loss_name][variant] = np.zeros(
                    (n_val, num_classes), dtype=np.float32
                )
                per_variant_per_loss_metrics[variant][loss_name] = {
                    "macro_auc": None, "per_class_auc": {},
                    "macro_f1": None, "kappa": None,
                    "sensitivity": {}, "specificity": {},
                    "thresholds": [], "best_epoch": 0,
                }

            # Memory hygiene between runs (guardrail #7)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Routing aggregation: per loss, soft-route 5 variant probs           #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Computing soft routing metrics per loss...")

    assert val_labels_shared is not None, "No run produced a valid eval result."

    routing_metrics: Dict[str, dict] = {}

    for loss_name in loss_experiments:
        prob_soft = soft_route_uniform(probs_by_loss_by_variant[loss_name], class_names)
        thr_soft  = find_optimal_thresholds(prob_soft, val_labels_shared, num_classes)
        met_soft  = compute_metrics(val_labels_shared, prob_soft, thr_soft, class_names)

        routing_metrics[loss_name] = {
            "soft_uniform": {
                "macro_auc":     met_soft["macro_auc"],
                "per_class_auc": met_soft["per_class_auc"],
                "macro_f1":      met_soft["macro_f1"],
                "kappa":         met_soft["kappa"],
                "sensitivity":   met_soft["sensitivity"],
                "specificity":   met_soft["specificity"],
                "thresholds":    thr_soft.tolist(),
            }
        }
        print(
            f"  {loss_name:<35} soft_auc={met_soft['macro_auc']:.4f}  "
            f"macro_f1={met_soft['macro_f1']:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Save combined JSON                                                  #
    # ------------------------------------------------------------------ #
    combined = {
        "experiment_name":            exp_name,
        "timestamp":                  datetime.utcnow().isoformat() + "Z",
        "config_snapshot":            cfg,
        "per_variant_per_loss_metrics": _to_serializable(per_variant_per_loss_metrics),
        "routing_metrics":            _to_serializable(routing_metrics),
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
    parser = argparse.ArgumentParser(
        description="Train V6: V1 × V2 (LSE+Aug) × Manifold Mixup × 5 losses"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 200 train / 50 val, 2 epochs per run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()

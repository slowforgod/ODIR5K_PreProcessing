"""
train/train_v2.py

Entry point for V2 (Data Augmentation + LSE oversampling) experiments.

Runs 5 loss functions sequentially on the same data/model configuration:
  1. bce_with_logits_loss
  2. focal_loss
  3. robust_asymmetric_loss
  4. cb_bce_with_logits_loss
  5. cb_focal_loss

Each loss gets a fresh ImageNet-pretrained ResNet-50 and its own output subfolder.

Usage:
    # Full training
    python -m train.train_v2 --config configs/v2.yaml

    # Smoke test (≤200 train + ≤50 val, 2 epochs per loss)
    python -m train.train_v2 --config configs/v2.yaml --smoke
"""

import argparse
import csv
import json
import os
import random
import sys
import warnings
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import yaml

warnings.filterwarnings("ignore")

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M"]


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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ODIRV2Dataset(Dataset):
    """ODIR-5K dataset for V2.

    Training mode: load image → V2Preprocessor.apply() → ToTensor + ImageNet normalize.
    Validation mode: load image → Resize → ToTensor + ImageNet normalize (no augmentation).

    The training dataset is built from an expanded record list (output of
    expand_with_lse); the validation dataset is built directly from val_df rows.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        records: List[dict],
        image_dir: str,
        img_size: int = 224,
        is_train: bool = True,
        preprocessor=None,
        smoke_n: int = 0,
    ):
        """
        Args:
            records     : list of {"filename": str, "labels": np.ndarray(7,), ...}
            image_dir   : path to preprocessed_images/
            img_size    : square resize target
            is_train    : if True, apply V2Preprocessor augmentation
            preprocessor: V2Preprocessor instance (used only when is_train=True)
            smoke_n     : if > 0, keep only first smoke_n records
        """
        if smoke_n > 0:
            records = records[:smoke_n]

        self.is_train = is_train
        self.preprocessor = preprocessor

        # Val-mode transform: resize → tensor → normalize
        self._val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
        ])

        # ToTensor + Normalize applied after augmentation in train mode
        self._to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
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
        img_pil = Image.open(img_path).convert("RGB")

        if self.is_train and self.preprocessor is not None:
            # V2Preprocessor operates on uint8 numpy (H, W, 3) RGB
            img_np = np.array(img_pil, dtype=np.uint8)
            img_np, _ = self.preprocessor.apply(img_np, label)
            # torchvision T.ToTensor accepts HWC uint8 numpy directly — no PIL roundtrip
            img_tensor = self._to_tensor_norm(img_np)
        else:
            img_tensor = self._val_transform(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Standard (non-Mixup) training epoch for V2."""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            # torch.amp.autocast (PyTorch 1.10+) — torch.cuda.amp.autocast deprecated in 2.0+
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v2.augmentation import V2Preprocessor
    from preprocessing.v2.oversampling import expand_with_lse
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
    from analysis.metrics import (
        compute_metrics,
        evaluate_with_tta,
    )

    # ------------------------------------------------------------------ #
    # Seed                                                                #
    # ------------------------------------------------------------------ #
    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

    # ------------------------------------------------------------------ #
    # Device + AMP                                                        #
    # ------------------------------------------------------------------ #
    requested = cfg["train"].get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA requested but not available — using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(requested if torch.cuda.is_available() else "cpu")

    use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"

    # ------------------------------------------------------------------ #
    # Data loading + splits                                               #
    # ------------------------------------------------------------------ #
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as f:
        split = json.load(f)

    train_ids = set(split[data_cfg["split_train_key"]])
    val_ids = set(split[data_cfg["split_val_key"]])

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[data_cfg["patient_id_col"]].isin(val_ids)].reset_index(drop=True)

    class_cols = data_cfg["class_cols"]
    image_dir = data_cfg["image_dir"]
    img_size = int(data_cfg.get("img_size", 224))

    # ------------------------------------------------------------------ #
    # CRITICAL: CB weights computed from ORIGINAL train_df (pre-LSE)     #
    # ------------------------------------------------------------------ #
    CLASS_WEIGHT_REFERENCE_DF = train_df.copy()
    labels_ref_np = CLASS_WEIGHT_REFERENCE_DF[class_cols].values.astype(np.float32)
    cb_beta = float(cfg.get("losses", {}).get("cb_beta", 0.9999))
    cb_normalize = cfg.get("losses", {}).get("cb_weight_normalize", "mean_one")
    cb_weights = effective_number_class_weights(
        labels_ref_np, beta=cb_beta, normalize=cb_normalize
    )
    print(f"[CB weights from original train_df ({len(train_df)} rows)]")
    for i, c in enumerate(class_cols):
        print(f"  {c}: {cb_weights[i]:.4f}")

    # ------------------------------------------------------------------ #
    # LSE expansion (training only)                                       #
    # ------------------------------------------------------------------ #
    pre_cfg = cfg.get("preprocessing", {})
    target_count = int(pre_cfg.get("lse_custom_target_count", 850))
    max_aug = int(pre_cfg.get("lse_max_aug_per_source", 12))

    expanded_records = expand_with_lse(
        train_df,
        class_cols=class_cols,
        target_count=target_count,
        max_aug_per_source=max_aug,
        seed=seed,
    )

    # Val records built directly from val_df rows
    val_records = [
        {
            "filename": str(row["filename"]),
            "labels": row[class_cols].values.astype(np.float32),
            "aug_idx": 0,
        }
        for _, row in val_df.iterrows()
    ]

    smoke_train = 200 if smoke else 0
    smoke_val = 50 if smoke else 0

    # ------------------------------------------------------------------ #
    # Preprocessor (augmentation — training mode only)                   #
    # ------------------------------------------------------------------ #
    # img_size는 data.img_size 단일 진실 소스 사용 — preprocessing 섹션에서 제외
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
    preprocessor = V2Preprocessor(img_size=img_size, **aug_kwargs)

    # ------------------------------------------------------------------ #
    # Datasets + DataLoaders                                              #
    # ------------------------------------------------------------------ #
    train_ds = ODIRV2Dataset(
        expanded_records,
        image_dir=image_dir,
        img_size=img_size,
        is_train=True,
        preprocessor=preprocessor,
        smoke_n=smoke_train,
    )
    val_ds = ODIRV2Dataset(
        val_records,
        image_dir=image_dir,
        img_size=img_size,
        is_train=False,
        preprocessor=None,
        smoke_n=smoke_val,
    )

    print(f"Train samples (expanded): {len(train_ds)} | Val samples: {len(val_ds)}")

    batch_size = cfg["train"].get("batch_size", 32)
    # Smoke 모드에선 워커 0으로 강제 (디버깅 단순화)
    num_workers = 0 if smoke else cfg["train"].get("num_workers", 4)

    # DO NOT use WeightedRandomSampler — LSE oversampling replaces it
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    # ------------------------------------------------------------------ #
    # Shared config                                                       #
    # ------------------------------------------------------------------ #
    num_epochs = 2 if smoke else cfg["train"].get("num_epochs", 20)
    patience = cfg["train"].get("patience", 8)
    lr = float(cfg["train"].get("lr", 1e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    dropout = float(cfg["train"].get("dropout", 0.5))
    num_classes = cfg.get("num_classes", 7)
    class_names = cfg.get("class_names", CLASS_NAMES)
    output_base = cfg.get("output_dir", "analysis/v2_analysis")
    exp_name = cfg.get("experiment_name", "v2_resnet50_seed42")

    loss_experiments = cfg.get("losses", {}).get(
        "experiments",
        ["bce_with_logits_loss", "focal_loss", "robust_asymmetric_loss",
         "cb_bce_with_logits_loss", "cb_focal_loss"],
    )
    loss_params = {k: cfg.get("losses", {}).get(k) for k in cfg.get("losses", {})}

    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    all_results = {}

    # ------------------------------------------------------------------ #
    # 5-loss training loop                                                #
    # ------------------------------------------------------------------ #
    for loss_idx, loss_name in enumerate(loss_experiments, start=1):
        print(f"\n{'='*60}")
        print(f"[Loss {loss_idx}/{len(loss_experiments)}: {loss_name}]")
        print(f"{'='*60}")

        # Per-loss output subfolder
        loss_dir = os.path.join(output_base, loss_name)
        os.makedirs(loss_dir, exist_ok=True)

        # ---- Fresh model for each loss ----
        seed_everything(seed)   # re-seed so each loss starts from identical weights
        model = build_model(
            name=cfg["model_name"],
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
        ).to(device)

        # ---- Criterion (move to device once — CB class_weights need to live on device) ----
        criterion = make_criterion(
            loss_name,
            cb_weights=cb_weights,
            params=loss_params,
        ).to(device)

        # ---- Adam optimizer (NOT AdamW — V2 spec) ----
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        # torch.amp.GradScaler (PyTorch 2.0+) — torch.cuda.amp.GradScaler deprecated
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ---- History CSV ----
        history_path = os.path.join(loss_dir, "history.csv")
        history_fields = [
            "epoch", "train_loss", "val_loss",
            "val_macro_auc", "val_macro_f1", "val_kappa",
        ]
        with open(history_path, "w", newline="") as hf:
            csv.DictWriter(hf, fieldnames=history_fields).writeheader()

        # ---- Training loop ----
        best_auc = -1.0
        no_improve = 0
        best_epoch = 0
        ckpt_path = os.path.join(loss_dir, "best.pth")

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )

            # Validation with horizontal-flip TTA (inline — same as V3 validate())
            model.eval()
            all_probs_list = []
            all_labels_list = []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    probs_orig = torch.sigmoid(model(imgs))
                    probs_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
                    probs_batch = (probs_orig + probs_flip) / 2.0
                    all_probs_list.append(probs_batch.cpu().numpy())
                    all_labels_list.append(labels.numpy())

            val_probs = np.concatenate(all_probs_list, axis=0)
            val_labels = np.concatenate(all_labels_list, axis=0)

            # val_loss: BCE of sigmoid probs vs true labels (monitoring only)
            val_loss = float(
                nn.BCELoss()(
                    torch.tensor(val_probs, dtype=torch.float32),
                    torch.tensor(val_labels, dtype=torch.float32),
                ).item()
            )

            # Per-epoch metrics at 0.5 thresholds (for logging; not final)
            epoch_metrics = compute_metrics(
                val_labels, val_probs, log_thresholds, class_names
            )
            val_auc = epoch_metrics["macro_auc"]
            val_f1 = epoch_metrics["macro_f1"]
            val_kappa = epoch_metrics["kappa"]

            scheduler.step()

            print(
                f"  Epoch {epoch:03d}/{num_epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"val_kappa={val_kappa:.4f}"
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

        # ---- Post-training: reload best.pth, TTA, thresholds ----
        if best_auc >= 0 and os.path.isfile(ckpt_path):
            print(f"\n  [reload] Loading best.pth (epoch {best_epoch}) and re-running TTA...")
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device)
            )

            eval_result = evaluate_with_tta(
                model, val_loader, device, num_classes,
                thresholds=None,
                class_names=class_names,
            )
            thresholds = eval_result["thresholds"]
            np.save(os.path.join(loss_dir, "thresholds.npy"), thresholds)

            loss_metrics = {
                "macro_auc":    eval_result["macro_auc"],
                "per_class_auc": eval_result["per_class_auc"],
                "macro_f1":     eval_result["macro_f1"],
                "kappa":        eval_result["kappa"],
                "sensitivity":  eval_result["sensitivity"],
                "specificity":  eval_result["specificity"],
                "thresholds":   thresholds.tolist(),
                "best_epoch":   best_epoch,
            }

            print(f"  macro_auc={loss_metrics['macro_auc']:.4f}  "
                  f"macro_f1={loss_metrics['macro_f1']:.4f}  "
                  f"kappa={loss_metrics['kappa']:.4f}")
            print(f"  thresholds → {loss_dir}/thresholds.npy")
        else:
            print(f"  [warn] No improvement recorded for {loss_name}.")
            loss_metrics = {
                "macro_auc": None, "per_class_auc": {},
                "macro_f1": None, "kappa": None,
                "sensitivity": {}, "specificity": {},
                "thresholds": [], "best_epoch": 0,
            }

        all_results[loss_name] = loss_metrics
        print(f"  history → {history_path}")

    # ------------------------------------------------------------------ #
    # Save combined JSON                                                  #
    # ------------------------------------------------------------------ #
    os.makedirs(output_base, exist_ok=True)

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

    combined = {
        "experiment_name": exp_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_snapshot": cfg,
        "results": _to_serializable(all_results),
    }

    json_path = os.path.join(output_base, f"{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All 5 losses complete.  Combined results → {json_path}")
    print("Summary:")
    for lname, lmet in all_results.items():
        auc = lmet.get("macro_auc")
        f1 = lmet.get("macro_f1")
        kappa = lmet.get("kappa")
        auc_s = f"{auc:.4f}" if auc is not None else "N/A"
        f1_s = f"{f1:.4f}" if f1 is not None else "N/A"
        kappa_s = f"{kappa:.4f}" if kappa is not None else "N/A"
        print(f"  {lname:<30} AUC={auc_s}  F1={f1_s}  Kappa={kappa_s}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 (Data Augmentation + LSE) model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 200 train / 50 val samples, 2 epochs per loss",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()

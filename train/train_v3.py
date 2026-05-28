"""
train/train_v3.py

Entry point for V3 (Manifold Mixup) training.

Usage:
    # Full training
    python -m train.train_v3 --config configs/v3.yaml

    # Smoke test (≤200 train + ≤50 val, 2 epochs)
    python -m train.train_v3 --config configs/v3.yaml --smoke
"""

import argparse
import csv
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = ["N", "D", "G", "C", "A", "H", "M"]


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

class ODIRDataset(Dataset):
    """Single-image dataset for ODIR-5K (7-class).

    Each row in the CSV corresponds to one jpg file.
    Images in data/preprocessed_images/ are already crop→CLAHE→resized.
    We just load, resize to ``img_size``, ToTensor, and ImageNet-normalise.
    """

    def __init__(
        self,
        df,
        image_dir: str,
        class_cols: list,
        img_size: int = 224,
        smoke_n: int = 0,
    ):
        """
        Args:
            df         : filtered pandas DataFrame (rows for this split)
            image_dir  : path to preprocessed_images/
            class_cols : label column names (7 classes)
            img_size   : square resize target (read from config.data.img_size)
            smoke_n    : if > 0, keep only first smoke_n rows
        """
        if smoke_n > 0:
            df = df.head(smoke_n)

        # Build transform with the requested img_size
        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Build valid-row list (skip missing images with a warning)
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
        img = Image.open(img_path).convert("RGB")
        img = self._transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: ODIRDataset) -> WeightedRandomSampler:
    """WeightedRandomSampler: sample_weight = sum of (1/class_freq) for each positive class."""
    labels = np.stack([s[1] for s in dataset.samples])   # (N, 7)
    class_freq = labels.sum(axis=0).clip(min=1)           # (7,)
    class_inv = 1.0 / class_freq
    sample_weights = torch.tensor(
        (labels * class_inv).sum(axis=1), dtype=torch.float32
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def make_pos_weight(dataset: ODIRDataset, device: torch.device) -> torch.Tensor:
    """pos_weight[i] = neg_count / pos_count for BCEWithLogitsLoss."""
    labels = np.stack([s[1] for s in dataset.samples])
    pos = labels.sum(axis=0).clip(min=1)
    neg = len(labels) - pos
    return torch.tensor(neg / pos, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Training / validation loops
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
):
    """One training epoch with Manifold Mixup."""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers)
                logits, label_a, label_b, lam = out
                loss = lam * criterion(logits, label_a) + (1.0 - lam) * criterion(logits, label_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers)
            logits, label_a, label_b, lam = out
            loss = lam * criterion(logits, label_a) + (1.0 - lam) * criterion(logits, label_b)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, num_classes: int):
    """Validation with horizontal-flip TTA. Returns probs and labels numpy arrays."""
    model.eval()
    all_probs = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits_orig = model(imgs)
        logits_flip = model(torch.flip(imgs, dims=[3]))
        probs = (torch.sigmoid(logits_orig) + torch.sigmoid(logits_flip)) / 2.0
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from analysis.metrics import (
        compute_metrics,
        evaluate_with_tta,
        save_metrics,
    )

    # --- seed ---
    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

    # --- device ---
    requested = cfg["train"].get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA requested but not available — using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(requested if torch.cuda.is_available() else "cpu")

    use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"

    # --- data ---
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as f:
        split = json.load(f)

    train_ids = set(split[data_cfg["split_train_key"]])
    val_ids = set(split[data_cfg["split_val_key"]])

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)]
    val_df = df[df[data_cfg["patient_id_col"]].isin(val_ids)]

    class_cols = data_cfg["class_cols"]
    image_dir = data_cfg["image_dir"]

    smoke_train = 200 if smoke else 0
    smoke_val = 50 if smoke else 0

    img_size = int(data_cfg.get("img_size", 224))

    train_ds = ODIRDataset(
        train_df, image_dir, class_cols, img_size=img_size, smoke_n=smoke_train,
    )
    val_ds = ODIRDataset(
        val_df, image_dir, class_cols, img_size=img_size, smoke_n=smoke_val,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    batch_size = cfg["train"].get("batch_size", 16)
    num_workers = cfg["train"].get("num_workers", 0) if not smoke else 0

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
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

    # --- model ---
    model = build_model(
        name=cfg["model_name"],
        num_classes=cfg.get("num_classes", 7),
        pretrained=True,
        dropout=cfg["train"].get("dropout", 0.5),
    ).to(device)

    # --- loss / optimiser / scheduler ---
    pos_weight = make_pos_weight(train_ds, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    num_epochs = 2 if smoke else cfg["train"].get("num_epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # --- Mixup params ---
    pre_cfg = cfg.get("preprocessing", {})
    alpha = float(pre_cfg.get("mixup_alpha", 0.2))
    mixup_layers = list(pre_cfg.get("mixup_layers", [0, 1, 2, 3]))

    # --- output dir ---
    output_dir = cfg.get("output_dir", "analysis/v3_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # --- history CSV ---
    history_path = os.path.join(output_dir, "history.csv")
    history_fields = ["epoch", "train_loss", "val_loss", "val_macro_auc", "val_macro_f1", "val_kappa"]
    with open(history_path, "w", newline="") as hf:
        csv.DictWriter(hf, fieldnames=history_fields).writeheader()

    # --- training loop ---
    best_auc = -1.0  # sentinel: any valid AUC (including 0.0) beats this
    no_improve = 0
    patience = cfg["train"].get("patience", 8)
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    # 0.5-threshold vector for per-epoch logging (compute_metrics needs thresholds).
    # Final reported metrics use F1-optimal thresholds (computed once after training).
    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    ckpt_path = os.path.join(output_dir, "best.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, alpha, mixup_layers, scaler
        )
        val_probs, val_labels = validate(model, val_loader, device, num_classes)

        # val_loss: BCE of sigmoid probabilities vs true labels (no pos_weight).
        # Used purely for monitoring; not a competition metric.
        val_loss = float(
            nn.BCELoss()(
                torch.tensor(val_probs, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.float32),
            ).item()
        )

        # ★ Single source of truth: analysis/metrics.compute_metrics ★
        # Uses 0.5 thresholds for the per-epoch snapshot. Final reported numbers
        # at the end of training use F1-optimal thresholds.
        epoch_metrics = compute_metrics(val_labels, val_probs, log_thresholds, class_names)
        val_auc = epoch_metrics["macro_auc"]
        val_f1 = epoch_metrics["macro_f1"]
        val_kappa = epoch_metrics["kappa"]

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_kappa={val_kappa:.4f}"
        )

        # Append to history CSV
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

        # Best model tracking
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [saved] best.pth (val_auc={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[early stop] no improvement for {patience} epochs")
                break

    # --- Post-training: reload best.pth, re-run TTA, compute thresholds + metrics ONCE ---
    if best_auc >= 0 and os.path.isfile(ckpt_path):
        print("\n[reload] Loading best.pth and re-running TTA evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # evaluate_with_tta(thresholds=None) runs TTA, then internally calls
        # find_optimal_thresholds(probs, labels) ONCE on the best val set,
        # then compute_metrics with those thresholds.
        eval_result = evaluate_with_tta(
            model, val_loader, device, num_classes,
            thresholds=None,
            class_names=class_names,
        )
        thresholds = eval_result["thresholds"]
        np.save(os.path.join(output_dir, "thresholds.npy"), thresholds)

        final_metrics = {
            k: eval_result[k]
            for k in (
                "macro_auc", "per_class_auc", "macro_f1",
                "kappa", "sensitivity", "specificity",
            )
        }
        exp_name = cfg.get("experiment_name", "v3_resnet50_seed42")
        json_path = save_metrics(
            final_metrics,
            save_dir=output_dir,
            experiment_name=exp_name,
            config_snapshot=cfg,
        )

        print(f"\n=== Final metrics (best epoch, reloaded) ===")
        print(f"  macro_auc  : {final_metrics['macro_auc']:.4f}")
        print(f"  macro_f1   : {final_metrics['macro_f1']:.4f}")
        print(f"  kappa      : {final_metrics['kappa']:.4f}")
        print(f"  thresholds saved → {output_dir}/thresholds.npy")
        print(f"  metrics JSON saved → {json_path}")
    else:
        print("[warn] No improvement was recorded — no thresholds/metrics saved.")

    print(f"\nHistory saved → {history_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 (Manifold Mixup) model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: 200 train / 50 val samples, 2 epochs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()

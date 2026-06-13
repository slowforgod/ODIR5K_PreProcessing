"""
train/train_v3.py

Entry point for V3 (Manifold Mixup) training.

Usage:
    python -m train_ResNet.train_v3 --config configs_ResNet/v3.yaml
    python -m train_ResNet.train_v3 --config configs_ResNet/v3.yaml --smoke

변경 이력:
  - split JSON: patient_split_7class_stratified.json (8:2)
              → patient_split_7class_stratified_811.json (8:1:1)
  - test split 로드 + test_loader 구성
  - BCEWithLogitsLoss → cb_bce_with_logits_loss 로 변경
  - 학습 후 test 평가 추가 (val threshold 재사용)
  - JSON을 val/test 분리 구조로 저장
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

class ODIRDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir: str,
        class_cols: list,
        img_size: int = 224,
        smoke_n: int = 0,
    ):
        if smoke_n > 0:
            df = df.head(smoke_n)

        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        img = Image.open(img_path).convert("RGB")
        img = self._transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: ODIRDataset) -> WeightedRandomSampler:
    labels = np.stack([s[1] for s in dataset.samples])
    class_freq = labels.sum(axis=0).clip(min=1)
    class_inv = 1.0 / class_freq
    sample_weights = torch.tensor(
        (labels * class_inv).sum(axis=1), dtype=torch.float32
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, alpha, mixup_layers, scaler=None):
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
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
def validate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
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
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
    from analysis.metrics import compute_metrics, evaluate_with_tta, save_metrics

    # ── Seed ────────────────────────────────────────────────────────────
    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

    # ── Device + AMP ────────────────────────────────────────────────────
    requested = cfg["train"].get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA requested but not available — using CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(requested if torch.cuda.is_available() else "cpu")

    use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"

    # ── Data loading + splits ★ test 추가 ───────────────────────────────
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])

    with open(data_cfg["split_path"]) as f:
        split = json.load(f)

    train_ids = set(split[data_cfg["split_train_key"]])
    val_ids   = set(split[data_cfg["split_val_key"]])
    test_key  = data_cfg.get("split_test_key")
    test_ids  = set(split[test_key]) if test_key and test_key in split else None

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)]
    val_df   = df[df[data_cfg["patient_id_col"]].isin(val_ids)]
    test_df  = (
        df[df[data_cfg["patient_id_col"]].isin(test_ids)].reset_index(drop=True)
        if test_ids is not None else None
    )

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 224))
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = len(class_names)

    smoke_train = 200 if smoke else 0
    smoke_val   = 50  if smoke else 0
    smoke_test  = 50  if smoke else 0

    train_ds = ODIRDataset(train_df, image_dir, class_cols, img_size=img_size, smoke_n=smoke_train)
    val_ds   = ODIRDataset(val_df,   image_dir, class_cols, img_size=img_size, smoke_n=smoke_val)
    test_ds  = (
        ODIRDataset(test_df, image_dir, class_cols, img_size=img_size, smoke_n=smoke_test)
        if test_df is not None else None
    )

    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}"
          + (f" / Test: {len(test_ds)}" if test_ds else ""))

    batch_size  = cfg["train"].get("batch_size", 32)
    num_workers = 0 if smoke else cfg["train"].get("num_workers", 4)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=(device.type == "cuda"))
        if test_ds is not None else None
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(
        name=cfg["model_name"],
        num_classes=num_classes,
        pretrained=True,
        dropout=cfg["train"].get("dropout", 0.5),
    ).to(device)

    # ── Loss: cb_bce_with_logits_loss ★ ─────────────────────────────────
    labels_np  = np.stack([s[1] for s in train_ds.samples])
    losses_cfg = cfg.get("losses", {})
    cb_beta    = float(losses_cfg.get("cb_beta", 0.9999))
    cb_norm    = losses_cfg.get("cb_weight_normalize", "mean_one")
    loss_name  = losses_cfg.get("loss_name", "cb_bce_with_logits_loss")

    cb_weights = effective_number_class_weights(labels_np, beta=cb_beta, normalize=cb_norm)
    print(f"[CB weights] loss={loss_name}")
    for i, c in enumerate(class_cols):
        print(f"  {c}: {cb_weights[i]:.4f}")

    criterion = make_criterion(loss_name, cb_weights=cb_weights, params=losses_cfg).to(device)

    # ── Optimizer / Scheduler ────────────────────────────────────────────
    num_epochs = 2 if smoke else cfg["train"].get("num_epochs", 20)
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Mixup params ─────────────────────────────────────────────────────
    pre_cfg      = cfg.get("preprocessing", {})
    alpha        = float(pre_cfg.get("mixup_alpha", 0.2))
    mixup_layers = list(pre_cfg.get("mixup_layers", [0, 1, 2, 3]))

    # ── Output ───────────────────────────────────────────────────────────
    output_dir = cfg.get("output_dir", "analysis/v3_analysis")
    os.makedirs(output_dir, exist_ok=True)

    history_path   = os.path.join(output_dir, "history.csv")
    history_fields = ["epoch", "train_loss", "val_loss", "val_macro_auc", "val_macro_f1", "val_kappa"]
    with open(history_path, "w", newline="") as hf:
        csv.DictWriter(hf, fieldnames=history_fields).writeheader()

    # ── Training loop ────────────────────────────────────────────────────
    best_auc  = -1.0
    no_improve = 0
    patience   = cfg["train"].get("patience", 8)
    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    ckpt_path  = os.path.join(output_dir, "best.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, alpha, mixup_layers, scaler
        )
        val_probs, val_labels = validate(model, val_loader, device)

        val_loss = float(nn.BCELoss()(
            torch.tensor(val_probs, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32),
        ).item())

        epoch_metrics = compute_metrics(val_labels, val_probs, log_thresholds, class_names)
        val_auc   = epoch_metrics["macro_auc"]
        val_f1    = epoch_metrics["macro_f1"]
        val_kappa = epoch_metrics["kappa"]
        scheduler.step()

        print(f"Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | "
              f"val_f1={val_f1:.4f} | val_kappa={val_kappa:.4f}")

        with open(history_path, "a", newline="") as hf:
            writer = csv.DictWriter(hf, fieldnames=history_fields)
            writer.writerow({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6), "val_macro_auc": round(val_auc, 6),
                "val_macro_f1": round(val_f1, 6), "val_kappa": round(val_kappa, 6),
            })

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

    # ── Post-training: val 평가 + threshold 최적화 ──────────────────────
    if best_auc >= 0 and os.path.isfile(ckpt_path):
        print("\n[reload] Loading best.pth and re-running TTA evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        val_result = evaluate_with_tta(
            model, val_loader, device, num_classes,
            thresholds=None, class_names=class_names,
        )
        thresholds = val_result["thresholds"]
        np.save(os.path.join(output_dir, "thresholds.npy"), thresholds)

        val_metrics = {k: val_result[k] for k in (
            "macro_auc", "per_class_auc", "macro_f1",
            "kappa", "sensitivity", "specificity",
        )}
        print(f"  [val]  auc={val_metrics['macro_auc']:.4f}  "
              f"f1={val_metrics['macro_f1']:.4f}  kappa={val_metrics['kappa']:.4f}")

        # ★ test 평가 — val threshold 재사용 (재계산 금지)
        test_metrics = None
        if test_loader is not None:
            test_result = evaluate_with_tta(
                model, test_loader, device, num_classes,
                thresholds=thresholds, class_names=class_names,
            )
            test_metrics = {k: test_result[k] for k in (
                "macro_auc", "per_class_auc", "macro_f1",
                "kappa", "sensitivity", "specificity",
            )}
            print(f"  [test] auc={test_metrics['macro_auc']:.4f}  "
                  f"f1={test_metrics['macro_f1']:.4f}  kappa={test_metrics['kappa']:.4f}")

        # ── 결과 저장 ────────────────────────────────────────────────────
        exp_name = cfg.get("experiment_name", "v3_resnet50_seed42")
        combined = {
            "experiment_name": exp_name,
            "split":           "8:1:1",
            "loss":            loss_name,
            "val":             _to_serializable(val_metrics),
            "test":            _to_serializable(test_metrics),
        }
        json_path = save_metrics(
            combined, save_dir=output_dir,
            experiment_name=exp_name, config_snapshot=cfg,
        )
        print(f"\n  thresholds → {output_dir}/thresholds.npy")
        print(f"  results    → {json_path}")
    else:
        print("[warn] No improvement recorded — no thresholds/metrics saved.")

    print(f"\nHistory → {history_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 (Manifold Mixup) model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 200 train / 50 val, 2 epochs")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
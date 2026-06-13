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
    python -m train_EfficientNet.train_v2 --config configs_EfficientNet/v2_efficientnet.yaml
    python -m train_EfficientNet.train_v2 --config configs_EfficientNet/v2_efficientnet.yaml --smoke

변경 이력:
  - split JSON: patient_split_7class_stratified.json (8:2)
              → patient_split_7class_stratified_811.json (8:1:1)
  - test split 로드 + test_loader 구성
  - 각 loss 학습 후 test 평가 추가 (val threshold 재사용)
  - 최종 JSON을 val/test 분리 구조로 저장
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

from preprocessing.common import crop_black_border as _crop_black_border

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

class ODIRV2Dataset(Dataset):
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
        crop_black_border: bool = False,
    ):
        if smoke_n > 0:
            records = records[:smoke_n]

        self.is_train = is_train
        self.preprocessor = preprocessor
        self.crop_black_border = crop_black_border

        self._val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
        ])
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
            print(f"  [warn] {missing} records skipped (image file not found)", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")

        # Optional black-border crop (off by default → ResNet V2 baseline unchanged).
        # Used by EfficientNet V2 to maximise effective resolution at 300×300.
        if self.crop_black_border:
            cropped = _crop_black_border(np.array(img_pil, dtype=np.uint8))
            img_pil = Image.fromarray(cropped)

        if self.is_train and self.preprocessor is not None:
            img_np = np.array(img_pil, dtype=np.uint8)
            img_np, _ = self.preprocessor.apply(img_np, label)
            img_tensor = self._to_tensor_norm(img_np)
        else:
            img_tensor = self._val_transform(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v2.augmentation import V2Preprocessor
    from preprocessing.v2.oversampling import expand_with_lse
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
    from analysis.metrics import compute_metrics, evaluate_with_tta

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

    train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[data_cfg["patient_id_col"]].isin(val_ids)].reset_index(drop=True)
    test_df  = (
        df[df[data_cfg["patient_id_col"]].isin(test_ids)].reset_index(drop=True)
        if test_ids is not None else None
    )

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 224))

    print(f"Train: {len(train_df)} / Val: {len(val_df)}"
          + (f" / Test: {len(test_df)}" if test_df is not None else ""))

    # ── CB weights (pre-LSE) ─────────────────────────────────────────────
    labels_ref_np = train_df[class_cols].values.astype(np.float32)
    cb_beta       = float(cfg.get("losses", {}).get("cb_beta", 0.9999))
    cb_normalize  = cfg.get("losses", {}).get("cb_weight_normalize", "mean_one")
    cb_weights    = effective_number_class_weights(labels_ref_np, beta=cb_beta, normalize=cb_normalize)
    print(f"[CB weights from original train_df ({len(train_df)} rows)]")
    for i, c in enumerate(class_cols):
        print(f"  {c}: {cb_weights[i]:.4f}")

    # ── LSE expansion ────────────────────────────────────────────────────
    pre_cfg      = cfg.get("preprocessing", {})
    target_count = int(pre_cfg.get("lse_custom_target_count", 850))
    max_aug      = int(pre_cfg.get("lse_max_aug_per_source", 12))

    expanded_records = expand_with_lse(
        train_df, class_cols=class_cols,
        target_count=target_count, max_aug_per_source=max_aug, seed=seed,
    )

    def _df_to_records(df):
        return [
            {"filename": str(row["filename"]),
             "labels": row[class_cols].values.astype(np.float32),
             "aug_idx": 0}
            for _, row in df.iterrows()
        ]

    val_records  = _df_to_records(val_df)
    test_records = _df_to_records(test_df) if test_df is not None else None

    smoke_train = 200 if smoke else 0
    smoke_val   = 50  if smoke else 0
    smoke_test  = 50  if smoke else 0

    # ── Preprocessor ─────────────────────────────────────────────────────
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

    crop_bb = bool(pre_cfg.get("crop_black_border", False))

    # ── Datasets + DataLoaders ───────────────────────────────────────────
    train_ds = ODIRV2Dataset(expanded_records, image_dir, img_size, is_train=True,
                              preprocessor=preprocessor, smoke_n=smoke_train,
                              crop_black_border=crop_bb)
    val_ds   = ODIRV2Dataset(val_records, image_dir, img_size, is_train=False,
                              preprocessor=None, smoke_n=smoke_val,
                              crop_black_border=crop_bb)

    print(f"Train samples (expanded): {len(train_ds)} | Val samples: {len(val_ds)}")

    batch_size  = cfg["train"].get("batch_size", 32)
    num_workers = 0 if smoke else cfg["train"].get("num_workers", 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ★ test_loader
    test_loader = None
    if test_records is not None:
        test_ds = ODIRV2Dataset(test_records, image_dir, img_size, is_train=False,
                                preprocessor=None, smoke_n=smoke_test,
                                crop_black_border=crop_bb)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=(device.type == "cuda"))
        print(f"Test samples: {len(test_ds)}")

    # ── Shared config ────────────────────────────────────────────────────
    num_epochs   = 2 if smoke else cfg["train"].get("num_epochs", 20)
    patience     = cfg["train"].get("patience", 8)
    lr           = float(cfg["train"].get("lr", 1e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    dropout      = float(cfg["train"].get("dropout", 0.5))
    num_classes  = cfg.get("num_classes", 7)
    class_names  = cfg.get("class_names", CLASS_NAMES)
    output_base  = cfg.get("output_dir", "analysis/v2_analysis")
    exp_name     = cfg.get("experiment_name", "v2_resnet50_seed42")

    loss_experiments = cfg.get("losses", {}).get(
        "experiments",
        ["bce_with_logits_loss", "focal_loss", "robust_asymmetric_loss",
         "cb_bce_with_logits_loss", "cb_focal_loss"],
    )
    loss_params = {k: cfg.get("losses", {}).get(k) for k in cfg.get("losses", {})}
    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    all_val_results  = {}
    all_test_results = {}

    # ── 5-loss training loop ─────────────────────────────────────────────
    for loss_idx, loss_name in enumerate(loss_experiments, start=1):
        print(f"\n{'='*60}")
        print(f"[Loss {loss_idx}/{len(loss_experiments)}: {loss_name}]")
        print(f"{'='*60}")

        loss_dir = os.path.join(output_base, loss_name)
        os.makedirs(loss_dir, exist_ok=True)

        seed_everything(seed)
        model = build_model(
            name=cfg["model_name"], num_classes=num_classes,
            pretrained=True, dropout=dropout,
        ).to(device)

        criterion = make_criterion(loss_name, cb_weights=cb_weights, params=loss_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler    = torch.amp.GradScaler("cuda") if use_amp else None

        history_path   = os.path.join(loss_dir, "history.csv")
        history_fields = ["epoch", "train_loss", "val_loss", "val_macro_auc", "val_macro_f1", "val_kappa"]
        with open(history_path, "w", newline="") as hf:
            csv.DictWriter(hf, fieldnames=history_fields).writeheader()

        best_f1    = -1.0
        no_improve = 0
        best_epoch = 0
        ckpt_path  = os.path.join(loss_dir, "best.pth")

        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

            model.eval()
            all_probs_list, all_labels_list = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    probs_orig = torch.sigmoid(model(imgs))
                    probs_flip = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
                    all_probs_list.append(((probs_orig + probs_flip) / 2.0).cpu().numpy())
                    all_labels_list.append(labels.numpy())

            val_probs  = np.concatenate(all_probs_list, axis=0)
            val_labels = np.concatenate(all_labels_list, axis=0)
            val_loss   = float(nn.BCELoss()(
                torch.tensor(val_probs, dtype=torch.float32),
                torch.tensor(val_labels, dtype=torch.float32),
            ).item())

            epoch_metrics = compute_metrics(val_labels, val_probs, log_thresholds, class_names)
            val_auc   = epoch_metrics["macro_auc"]          # history.csv 기록용
            val_f1    = epoch_metrics["macro_f1"]
            val_kappa = epoch_metrics["kappa"]
            val_sens  = float(np.mean(list(epoch_metrics["sensitivity"].values())))
            val_spec  = float(np.mean(list(epoch_metrics["specificity"].values())))
            scheduler.step()

            print(f"  Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                  f"val_kappa={val_kappa:.4f} | val_sens={val_sens:.4f} | val_spec={val_spec:.4f}")

            with open(history_path, "a", newline="") as hf:
                writer = csv.DictWriter(hf, fieldnames=history_fields)
                writer.writerow({
                    "epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6), "val_macro_auc": round(val_auc, 6),
                    "val_macro_f1": round(val_f1, 6), "val_kappa": round(val_kappa, 6),
                })

            if not np.isnan(val_f1) and val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"    [saved] best.pth (val_f1={best_f1:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [early stop] no improvement for {patience} epochs")
                    break

        # ── Post-training: val 평가 + threshold 최적화 ──────────────────
        if best_f1 >= 0 and os.path.isfile(ckpt_path):
            print(f"\n  [reload] Loading best.pth (epoch {best_epoch}) ...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            val_result = evaluate_with_tta(
                model, val_loader, device, num_classes,
                thresholds=None, class_names=class_names,
            )
            thresholds = val_result["thresholds"]
            np.save(os.path.join(loss_dir, "thresholds.npy"), thresholds)

            all_val_results[loss_name] = {
                "macro_auc":     val_result["macro_auc"],
                "per_class_auc": val_result["per_class_auc"],
                "macro_f1":      val_result["macro_f1"],
                "kappa":         val_result["kappa"],
                "sensitivity":   val_result["sensitivity"],
                "specificity":   val_result["specificity"],
                "thresholds":    thresholds.tolist(),
                "best_epoch":    best_epoch,
            }
            v_sens = float(np.mean(list(val_result["sensitivity"].values())))
            v_spec = float(np.mean(list(val_result["specificity"].values())))
            print(f"  [val]  f1={val_result['macro_f1']:.4f}  kappa={val_result['kappa']:.4f}  "
                  f"sens={v_sens:.4f}  spec={v_spec:.4f}")

            # ★ test 평가 — val threshold 재사용 (재계산 금지)
            if test_loader is not None:
                test_result = evaluate_with_tta(
                    model, test_loader, device, num_classes,
                    thresholds=thresholds, class_names=class_names,
                )
                all_test_results[loss_name] = {
                    "macro_auc":     test_result["macro_auc"],
                    "per_class_auc": test_result["per_class_auc"],
                    "macro_f1":      test_result["macro_f1"],
                    "kappa":         test_result["kappa"],
                    "sensitivity":   test_result["sensitivity"],
                    "specificity":   test_result["specificity"],
                }
                t_sens = float(np.mean(list(test_result["sensitivity"].values())))
                t_spec = float(np.mean(list(test_result["specificity"].values())))
                print(f"  [test] f1={test_result['macro_f1']:.4f}  kappa={test_result['kappa']:.4f}  "
                      f"sens={t_sens:.4f}  spec={t_spec:.4f}")
        else:
            print(f"  [warn] No improvement recorded for {loss_name}.")
            all_val_results[loss_name] = {
                "macro_auc": None, "per_class_auc": {}, "macro_f1": None,
                "kappa": None, "sensitivity": {}, "specificity": {},
                "thresholds": [], "best_epoch": 0,
            }
            if test_loader is not None:
                all_test_results[loss_name] = {
                    "macro_auc": None, "per_class_auc": {}, "macro_f1": None,
                    "kappa": None, "sensitivity": {}, "specificity": {},
                }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 결과 저장 ────────────────────────────────────────────────────────
    os.makedirs(output_base, exist_ok=True)

    combined = {
        "experiment_name": exp_name,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "config_snapshot": cfg,
        "split":           "8:1:1",
        "val":             _to_serializable(all_val_results),
        "test":            _to_serializable(all_test_results) if test_loader is not None else None,
    }

    json_path = os.path.join(output_base, f"{exp_name}.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All {len(loss_experiments)} losses complete.  Results → {json_path}")
    print("Summary (val):")
    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "N/A"
    def _mean(d):
        return float(np.mean(list(d.values()))) if isinstance(d, dict) and d else None
    for lname, lmet in all_val_results.items():
        f1   = lmet.get("macro_f1")
        k    = lmet.get("kappa")
        sens = _mean(lmet.get("sensitivity"))
        spec = _mean(lmet.get("specificity"))
        print(f"  {lname:<30} F1={_fmt(f1)}  Kappa={_fmt(k)}  Sens={_fmt(sens)}  Spec={_fmt(spec)}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 (Data Augmentation + LSE) model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 200 train / 50 val, 2 epochs per loss")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
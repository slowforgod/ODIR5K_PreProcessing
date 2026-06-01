"""
train/train_v4.py

Entry point for V4: V1 soft routing × V2 (LSE + Augmentation) × 1 loss (cb_bce_with_logits_loss).

Outer loop: 4 CLAHE variants (V1_L, V1_LA, V1_LB, V1_LAB)
Inner loop: 1 loss function (cb_bce_with_logits_loss)

Total: 4 variants × 1 loss = 4 training runs.
After all runs, soft_route_uniform is applied across 4 variant probs.

Usage:
    python -m train.train_v4 --config configs/v4.yaml
    python -m train.train_v4 --config configs/v4.yaml --smoke

변경 이력:
  - test split 로드 + test_loader 구성
  - 각 variant/loss 학습 후 test 평가 추가 (val threshold 재사용)
  - soft routing test 평가 추가
  - JSON을 val/test 분리 구조로 저장
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

class ODIRV4TrainDataset(Dataset):
    def __init__(self, records, image_dir, img_size=224, preprocessor=None, smoke_n=0):
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
            print(f"  [warn] {missing} records skipped", file=sys.stderr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_pil = self._resize(Image.open(img_path).convert("RGB"))
        img_np = np.array(img_pil, dtype=np.uint8)
        if self.preprocessor is not None:
            img_np, _ = self.preprocessor.apply(img_np, label)
        return self._to_tensor_norm(img_np), torch.tensor(label, dtype=torch.float32)


class ODIRV4EvalDataset(Dataset):
    """Val/Test dataset: V1 CLAHE only, no V2 augmentation."""
    def __init__(self, records, image_dir, img_size=224, preprocessor=None, smoke_n=0):
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
            print(f"  [warn] {missing} records skipped", file=sys.stderr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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
    from preprocessing.v1.clahe import V1Preprocessor
    from preprocessing.v1.routing import DEFAULT_VARIANTS, soft_route_uniform
    from preprocessing.v2.augmentation import V2Preprocessor
    from preprocessing.v2.oversampling import expand_with_lse
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
    from preprocessing.v4.preprocess import V4Preprocessor
    from analysis.metrics import compute_metrics, evaluate_with_tta, find_optimal_thresholds

    # ── Seed + device ────────────────────────────────────────────────────
    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

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

    with open(data_cfg["split_path"]) as fh:
        split_json = json.load(fh)

    train_ids = set(split_json[data_cfg["split_train_key"]])
    val_ids   = set(split_json[data_cfg["split_val_key"]])
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

    print(f"Train: {len(train_df)} / Val: {len(val_df)}"
          + (f" / Test: {len(test_df)}" if test_df is not None else ""))

    # ── CB weights (pre-LSE) ─────────────────────────────────────────────
    labels_ref_np = train_df[class_cols].values.astype(np.float32)
    cb_beta       = float(cfg.get("losses", {}).get("cb_beta", 0.9999))
    cb_normalize  = cfg.get("losses", {}).get("cb_weight_normalize", "mean_one")
    cb_mode       = cfg.get("losses", {}).get("cb_weight_mode", "effective_number")
    cb_weights    = effective_number_class_weights(
        labels_ref_np, beta=cb_beta, normalize=cb_normalize, mode=cb_mode,
    )
    print(f"[CB weights, mode={cb_mode}]")
    for i, c in enumerate(class_cols):
        print(f"  {c}: w={cb_weights[i]:.4f}")

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

    # ── Config ───────────────────────────────────────────────────────────
    variants    = pre_cfg.get("variants", DEFAULT_VARIANTS)
    clip_limit  = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid   = tuple(pre_cfg.get("tile_grid_size", [8, 8]))

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

    loss_experiments = cfg.get("losses", {}).get("experiments", ["cb_bce_with_logits_loss"])
    loss_params      = {k: cfg.get("losses", {}).get(k) for k in cfg.get("losses", {})}

    output_base = cfg.get("output_dir", "analysis/v4_analysis")
    exp_name    = cfg.get("experiment_name", "v4_resnet50_seed42")
    os.makedirs(output_base, exist_ok=True)

    log_thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    probs_val_by_loss_by_variant:  Dict[str, Dict[str, np.ndarray]] = {ln: {} for ln in loss_experiments}
    probs_test_by_loss_by_variant: Dict[str, Dict[str, np.ndarray]] = {ln: {} for ln in loss_experiments}
    per_variant_per_loss_val:  Dict[str, Dict[str, dict]] = {}
    per_variant_per_loss_test: Dict[str, Dict[str, dict]] = {}
    val_labels_shared:  Optional[np.ndarray] = None
    test_labels_shared: Optional[np.ndarray] = None

    # ── Outer loop: variants ─────────────────────────────────────────────
    for v_idx, variant in enumerate(variants, start=1):
        print(f"\n{'='*60}")
        print(f"[Variant {v_idx}/{len(variants)}: {variant}]")
        print(f"{'='*60}")

        per_variant_per_loss_val[variant]  = {}
        per_variant_per_loss_test[variant] = {}

        v1_pre = V1Preprocessor(variant=variant, clip_limit=clip_limit, tile_grid_size=tile_grid)
        v4_pre = V4Preprocessor(
            v1_kwargs=dict(variant=variant, clip_limit=clip_limit, tile_grid_size=tile_grid),
            v2_kwargs=dict(img_size=img_size, **aug_kwargs),
        )

        train_ds = ODIRV4TrainDataset(expanded_records, image_dir, img_size, v4_pre, smoke_train)
        val_ds   = ODIRV4EvalDataset(val_records, image_dir, img_size, v1_pre, smoke_val)
        test_ds  = (
            ODIRV4EvalDataset(test_records, image_dir, img_size, v1_pre, smoke_test)
            if test_records is not None else None
        )

        print(f"  Train (expanded): {len(train_ds)}  Val: {len(val_ds)}"
              + (f"  Test: {len(test_ds)}" if test_ds else ""))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=(device.type == "cuda"))
        test_loader  = (
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=(device.type == "cuda"))
            if test_ds is not None else None
        )

        # ── Inner loop: losses ───────────────────────────────────────────
        for loss_idx, loss_name in enumerate(loss_experiments, start=1):
            print(f"\n  [{variant}] Loss {loss_idx}/{len(loss_experiments)}: {loss_name}")

            loss_dir = os.path.join(output_base, variant, loss_name)
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

            best_auc   = -1.0
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

                epoch_metrics  = compute_metrics(val_labels, val_probs, log_thresholds, class_names)
                val_auc        = epoch_metrics["macro_auc"]
                val_f1         = epoch_metrics["macro_f1"]
                val_kappa      = epoch_metrics["kappa"]
                sens_per_class = epoch_metrics["sensitivity"]
                val_sens       = float(np.mean(list(sens_per_class.values())))
                scheduler.step()

                sens_str = " ".join(f"{c}={sens_per_class.get(c, float('nan')):.3f}" for c in class_names)
                print(f"    Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                      f"val_sens_macro={val_sens:.4f} | sens[{sens_str}]")

                with open(history_path, "a", newline="") as hf:
                    writer = csv.DictWriter(hf, fieldnames=history_fields)
                    writer.writerow({
                        "epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6), "val_macro_auc": round(val_auc, 6),
                        "val_macro_f1": round(val_f1, 6), "val_kappa": round(val_kappa, 6),
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

            # ── Post-training: val 평가 + threshold 최적화 ──────────────
            if best_auc >= 0 and os.path.isfile(ckpt_path):
                print(f"\n    [reload] Loading best.pth (epoch {best_epoch}) ...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))

                val_result = evaluate_with_tta(
                    model, val_loader, device, num_classes,
                    thresholds=None, class_names=class_names,
                )
                thresholds = val_result["thresholds"]
                np.save(os.path.join(loss_dir, "thresholds.npy"), thresholds)

                probs_val_by_loss_by_variant[loss_name][variant] = val_result["probs"]
                if val_labels_shared is None:
                    val_labels_shared = val_result["labels"]

                per_variant_per_loss_val[variant][loss_name] = {
                    "macro_auc": val_result["macro_auc"], "per_class_auc": val_result["per_class_auc"],
                    "macro_f1": val_result["macro_f1"], "kappa": val_result["kappa"],
                    "sensitivity": val_result["sensitivity"], "specificity": val_result["specificity"],
                    "thresholds": thresholds.tolist(), "best_epoch": best_epoch,
                }
                print(f"    [val]  auc={val_result['macro_auc']:.4f}  "
                      f"f1={val_result['macro_f1']:.4f}  kappa={val_result['kappa']:.4f}")

                # ★ test 평가 — val threshold 재사용
                if test_loader is not None:
                    test_result = evaluate_with_tta(
                        model, test_loader, device, num_classes,
                        thresholds=thresholds, class_names=class_names,
                    )
                    probs_test_by_loss_by_variant[loss_name][variant] = test_result["probs"]
                    if test_labels_shared is None:
                        test_labels_shared = test_result["labels"]

                    per_variant_per_loss_test[variant][loss_name] = {
                        "macro_auc": test_result["macro_auc"], "per_class_auc": test_result["per_class_auc"],
                        "macro_f1": test_result["macro_f1"], "kappa": test_result["kappa"],
                        "sensitivity": test_result["sensitivity"], "specificity": test_result["specificity"],
                    }
                    print(f"    [test] auc={test_result['macro_auc']:.4f}  "
                          f"f1={test_result['macro_f1']:.4f}  kappa={test_result['kappa']:.4f}")
            else:
                print(f"    [warn] No improvement for {variant}/{loss_name}.")
                probs_val_by_loss_by_variant[loss_name][variant]  = np.zeros((len(val_ds), num_classes), dtype=np.float32)
                per_variant_per_loss_val[variant][loss_name]  = {
                    "macro_auc": None, "per_class_auc": {}, "macro_f1": None,
                    "kappa": None, "sensitivity": {}, "specificity": {},
                    "thresholds": [], "best_epoch": 0,
                }
                if test_loader is not None:
                    probs_test_by_loss_by_variant[loss_name][variant] = np.zeros((len(test_ds), num_classes), dtype=np.float32)
                    per_variant_per_loss_test[variant][loss_name] = {
                        "macro_auc": None, "per_class_auc": {}, "macro_f1": None,
                        "kappa": None, "sensitivity": {}, "specificity": {},
                    }

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Routing aggregation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Computing soft routing metrics per loss...")
    assert val_labels_shared is not None, "No run produced a valid eval result."

    val_routing_metrics:  Dict[str, dict] = {}
    test_routing_metrics: Dict[str, dict] = {}

    for loss_name in loss_experiments:
        # val soft routing
        prob_soft_val = soft_route_uniform(probs_val_by_loss_by_variant[loss_name], class_names)
        thr_soft      = find_optimal_thresholds(prob_soft_val, val_labels_shared, num_classes)
        met_soft_val  = compute_metrics(val_labels_shared, prob_soft_val, thr_soft, class_names)
        val_routing_metrics[loss_name] = {
            "soft_uniform": {**met_soft_val, "thresholds": thr_soft.tolist()}
        }
        print(f"  [val]  {loss_name:<30} soft_auc={met_soft_val['macro_auc']:.4f}  f1={met_soft_val['macro_f1']:.4f}")

        # test soft routing — val threshold 재사용
        if test_labels_shared is not None:
            prob_soft_test = soft_route_uniform(probs_test_by_loss_by_variant[loss_name], class_names)
            met_soft_test  = compute_metrics(test_labels_shared, prob_soft_test, thr_soft, class_names)
            test_routing_metrics[loss_name] = {"soft_uniform": met_soft_test}
            print(f"  [test] {loss_name:<30} soft_auc={met_soft_test['macro_auc']:.4f}  f1={met_soft_test['macro_f1']:.4f}")

    # ── 결과 저장 ────────────────────────────────────────────────────────
    combined = {
        "experiment_name": exp_name,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "config_snapshot": cfg,
        "split":           "8:1:1",
        "val": {
            "per_variant_per_loss_metrics": _to_serializable(per_variant_per_loss_val),
            "routing_metrics":              _to_serializable(val_routing_metrics),
        },
        "test": {
            "per_variant_per_loss_metrics": _to_serializable(per_variant_per_loss_test),
            "routing_metrics":              _to_serializable(test_routing_metrics),
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
    parser = argparse.ArgumentParser(description="Train V4: V1 × V2 × cb_bce_with_logits_loss")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 200 train / 50 val, 2 epochs per run")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
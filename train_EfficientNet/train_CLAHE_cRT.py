"""
train_EfficientNet/train_CLAHE_cRT.py

Task B — CLAHE + cRT (classifier Re-Training, 2-stage) on EfficientNet-B3.

Base = V1 soft-routing ensemble: for each CLAHE variant (V1_L/LA/LB/LAB):
    stage-1 : full training (backbone + head), V1's normal loss (weighted_bce).
    stage-2 : freeze backbone (requires_grad=False, BN in eval), re-initialize
              the classifier head (nn.Linear(1536, 7)), and re-train ONLY the
              head for a few epochs with an inverse-frequency WeightedRandomSampler
              (soft class balancing — NO hard single-label filtering, so
              multi-label co-occurrence is preserved).
Then soft-route (uniform average) the per-variant cRT'd probabilities.

NOTE on sampling: both stages use the inverse-freq WeightedRandomSampler (V1's
existing setup, per "stage-1은 기존 학습 그대로"). Textbook cRT decouples
sampling (stage-1 instance-balanced, stage-2 class-balanced); to do that, swap
stage-1 to a plain shuffle loader. Left as the literal spec for now.

Pipeline per image (train & eval identical — V1 applies CLAHE to both):
    open → crop_black_border → CLAHE(variant) → resize(img_size) → normalize

Selection: macro-F1 best-epoch (both stages). Threshold: per-class F1-max search
(evaluate_with_tta → find_optimal_thresholds), saved as thresholds.npy.

Usage:
    python -m train_EfficientNet.train_CLAHE_cRT --config configs_EfficientNet/CLAHE_cRT.yaml
    python -m train_EfficientNet.train_CLAHE_cRT --config configs_EfficientNet/CLAHE_cRT.yaml --smoke
"""

import argparse
import csv
import json
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import yaml

from preprocessing.common import crop_black_border

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


def _build_sampler(dataset):
    """WeightedRandomSampler with inverse class-frequency weights, aligned to
    dataset.samples (after smoke truncation / missing-file filtering)."""
    labels = np.stack([lbl for _, lbl in dataset.samples]).astype(np.float32)
    class_freq = labels.mean(axis=0)
    inv_freq = 1.0 / np.clip(class_freq, 1e-6, None)
    sample_weights = (labels * inv_freq[None, :]).sum(axis=1)
    sample_weights = np.clip(sample_weights, 1e-6, None)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Dataset — CLAHE variant applied to all splits (no augmentation)
# ---------------------------------------------------------------------------

class ODIRClaheDataset(Dataset):
    def __init__(self, records, image_dir, img_size, preprocessor, smoke_n=0):
        if smoke_n > 0:
            records = records[:smoke_n]
        self.preprocessor = preprocessor
        self.img_size = img_size
        self._to_tensor_norm = T.Compose([
            T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
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
            print(f"  [warn] {missing} records skipped (image not found)", file=sys.stderr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img_np = crop_black_border(img_np)
        img_np, _ = self.preprocessor.apply(img_np, label)   # CLAHE
        img_pil = Image.fromarray(img_np).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        return self._to_tensor_norm(img_pil), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# One training stage (shared by stage-1 and stage-2)
# ---------------------------------------------------------------------------

def run_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
              device, num_epochs, patience, scaler, grad_clip, class_names,
              num_classes, ckpt_path, history_path, freeze_backbone, tag):
    """Train `num_epochs`, select best epoch by macro-F1, save to ckpt_path.

    If freeze_backbone is True, model.features is kept in eval() so BN running
    stats stay frozen while only the head trains.
    """
    from analysis.metrics import compute_metrics
    log_thr = np.full(num_classes, 0.5, dtype=np.float32)
    history_fields = ["epoch", "train_loss", "val_macro_auc", "val_macro_f1", "val_kappa"]
    with open(history_path, "w", newline="") as hf:
        csv.DictWriter(hf, fieldnames=history_fields).writeheader()

    best_f1, no_improve, best_epoch = -1.0, 0, 0
    use_amp = scaler is not None

    for epoch in range(1, num_epochs + 1):
        model.train()
        if freeze_backbone:
            model.features.eval()   # freeze BN running stats in the backbone
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), labels)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        pl, ll = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                p = (torch.sigmoid(model(imgs)) +
                     torch.sigmoid(model(torch.flip(imgs, dims=[3])))) / 2.0
                pl.append(p.cpu().numpy()); ll.append(labels.numpy())
        vp, vl = np.concatenate(pl), np.concatenate(ll)
        m = compute_metrics(vl, vp, log_thr, class_names)
        val_auc, val_f1, val_kappa = m["macro_auc"], m["macro_f1"], m["kappa"]
        val_sens = float(np.mean(list(m["sensitivity"].values())))
        val_spec = float(np.mean(list(m["specificity"].values())))
        if scheduler is not None:
            scheduler.step()

        print(f"    [{tag}] Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.4f} | "
              f"val_f1={val_f1:.4f} | val_kappa={val_kappa:.4f} | "
              f"val_sens={val_sens:.4f} | val_spec={val_spec:.4f}")
        with open(history_path, "a", newline="") as hf:
            csv.DictWriter(hf, fieldnames=history_fields).writerow({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_macro_auc": round(val_auc, 6), "val_macro_f1": round(val_f1, 6),
                "val_kappa": round(val_kappa, 6)})

        if not np.isnan(val_f1) and val_f1 > best_f1:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"        [saved] {os.path.basename(ckpt_path)} (val_f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{tag}] early stop (no improvement for {patience} epochs)")
                break
    return best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    from model.base import build_model
    from preprocessing.v1.clahe import V1Preprocessor
    from preprocessing.v1.routing import soft_route_uniform
    from analysis.metrics import compute_metrics, evaluate_with_tta, find_optimal_thresholds

    seed = cfg["train"].get("seed", 42)
    seed_everything(seed)

    requested = cfg["train"].get("device", "cpu")
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("use_amp", False) and device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}")

    # ── Data + splits ────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["csv_path"])
    with open(data_cfg["split_path"]) as fh:
        split_json = json.load(fh)

    train_ids = set(split_json[data_cfg["split_train_key"]])
    val_ids   = set(split_json[data_cfg["split_val_key"]])
    test_key  = data_cfg.get("split_test_key")
    test_ids  = set(split_json[test_key]) if test_key and test_key in split_json else None

    pid = data_cfg["patient_id_col"]
    train_df = df[df[pid].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df[pid].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df[pid].isin(test_ids)].reset_index(drop=True) if test_ids is not None else None

    class_cols  = data_cfg["class_cols"]
    image_dir   = data_cfg["image_dir"]
    img_size    = int(data_cfg.get("img_size", 300))
    class_names = cfg.get("class_names", CLASS_NAMES)
    num_classes = cfg.get("num_classes", 7)

    def _records(d):
        return [{"filename": str(r["filename"]),
                 "labels": r[class_cols].values.astype(np.float32)}
                for _, r in d.iterrows()]

    train_records = _records(train_df)
    val_records   = _records(val_df)
    test_records  = _records(test_df) if test_df is not None else None

    print(f"Train: {len(train_records)} / Val: {len(val_records)}"
          + (f" / Test: {len(test_records)}" if test_records else ""))

    # ── weighted_bce pos_weight (stage-1, from train split) ──────────────
    pos = train_df[class_cols].sum(axis=0).values.astype(np.float32)
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32).to(device)

    # ── Config ───────────────────────────────────────────────────────────
    pre_cfg    = cfg.get("preprocessing", {})
    variants   = pre_cfg.get("variants", ["V1_L", "V1_LA", "V1_LB", "V1_LAB"])
    clip_limit = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid  = tuple(pre_cfg.get("tile_grid_size", [8, 8]))

    tcfg         = cfg["train"]
    batch_size   = int(tcfg.get("batch_size", 32))
    num_epochs   = 2 if smoke else int(tcfg.get("num_epochs", 30))
    patience     = int(tcfg.get("patience", 8))
    lr           = float(tcfg.get("lr", 1e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    dropout      = float(tcfg.get("dropout", 0.4))
    num_workers  = 0 if smoke else int(tcfg.get("num_workers", 4))
    grad_clip    = float(tcfg.get("grad_clip", 0.0))
    stage1_loss  = str(tcfg.get("stage1_loss", "weighted_bce"))

    crt_cfg      = cfg.get("crt", {})
    head_epochs  = 2 if smoke else int(crt_cfg.get("head_epochs", 8))
    head_lr      = float(crt_cfg.get("head_lr", 1e-3))
    head_wd      = float(crt_cfg.get("head_weight_decay", 1e-4))

    smoke_n = 200 if smoke else 0
    smoke_v = 50  if smoke else 0

    output_base = cfg.get("output_dir", "analysis/EfficientNet/CLAHE_cRT_analysis")
    exp_name    = cfg.get("experiment_name", "CLAHE_cRT_efficientnet_seed42")
    os.makedirs(output_base, exist_ok=True)

    if stage1_loss != "weighted_bce":
        raise ValueError(f"stage1_loss '{stage1_loss}' not supported; use 'weighted_bce'.")

    probs_val_by_variant:  Dict[str, np.ndarray] = {}
    probs_test_by_variant: Dict[str, np.ndarray] = {}
    per_variant_val:  Dict[str, dict] = {}
    per_variant_test: Dict[str, dict] = {}
    val_labels_shared:  Optional[np.ndarray] = None
    test_labels_shared: Optional[np.ndarray] = None

    # ── Variant loop ─────────────────────────────────────────────────────
    for v_idx, variant in enumerate(variants, start=1):
        print(f"\n{'='*60}\n[Variant {v_idx}/{len(variants)}: {variant}]\n{'='*60}")
        v_dir = os.path.join(output_base, variant)
        os.makedirs(v_dir, exist_ok=True)

        pre = V1Preprocessor(variant=variant, clip_limit=clip_limit, tile_grid_size=tile_grid)
        train_ds = ODIRClaheDataset(train_records, image_dir, img_size, pre, smoke_n)
        val_ds   = ODIRClaheDataset(val_records, image_dir, img_size, pre, smoke_v)
        test_ds  = ODIRClaheDataset(test_records, image_dir, img_size, pre, smoke_v) if test_records else None

        sampler = _build_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=(device.type == "cuda"))
        test_loader  = (DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=(device.type == "cuda"))
                        if test_ds else None)

        seed_everything(seed)
        model = build_model(name=cfg["model_name"], num_classes=num_classes,
                            pretrained=True, dropout=dropout).to(device)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── Stage 1: full training ───────────────────────────────────────
        print("  --- Stage 1: full training ---")
        criterion1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim1 = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=num_epochs)
        s1_ckpt = os.path.join(v_dir, "stage1_best.pth")
        s1_epoch = run_stage(model, train_loader, val_loader, criterion1, optim1, sched1,
                             device, num_epochs, patience, scaler, grad_clip, class_names,
                             num_classes, s1_ckpt, os.path.join(v_dir, "history_stage1.csv"),
                             freeze_backbone=False, tag="S1")
        model.load_state_dict(torch.load(s1_ckpt, map_location=device))

        # ── Stage 2: cRT — freeze backbone, reinit head, retrain head ────
        print("  --- Stage 2: cRT (freeze backbone, re-train head) ---")
        for p in model.features.parameters():
            p.requires_grad = False
        model.classifier[-1].reset_parameters()    # re-init nn.Linear(1536, 7)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optim2 = torch.optim.Adam(head_params, lr=head_lr, weight_decay=head_wd)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=head_epochs)
        scaler2 = torch.amp.GradScaler("cuda") if use_amp else None
        ckpt_path = os.path.join(v_dir, "best.pth")     # cRT'd head = final model
        s2_epoch = run_stage(model, train_loader, val_loader, criterion1, optim2, sched2,
                             device, head_epochs, patience, scaler2, grad_clip, class_names,
                             num_classes, ckpt_path, os.path.join(v_dir, "history_stage2.csv"),
                             freeze_backbone=True, tag="S2")

        # ── Post-training: reload best (cRT) → threshold opt + test ──────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        val_res = evaluate_with_tta(model, val_loader, device, num_classes,
                                    thresholds=None, class_names=class_names)
        thr = val_res["thresholds"]
        np.save(os.path.join(v_dir, "thresholds.npy"), thr)
        probs_val_by_variant[variant] = val_res["probs"]
        if val_labels_shared is None:
            val_labels_shared = val_res["labels"]
        per_variant_val[variant] = {
            "macro_auc": val_res["macro_auc"], "per_class_auc": val_res["per_class_auc"],
            "macro_f1": val_res["macro_f1"], "kappa": val_res["kappa"],
            "sensitivity": val_res["sensitivity"], "specificity": val_res["specificity"],
            "thresholds": thr.tolist(),
            "stage1_best_epoch": s1_epoch, "stage2_best_epoch": s2_epoch,
        }
        v_sens = float(np.mean(list(val_res["sensitivity"].values())))
        v_spec = float(np.mean(list(val_res["specificity"].values())))
        print(f"  [val]  f1={val_res['macro_f1']:.4f}  kappa={val_res['kappa']:.4f}  "
              f"sens={v_sens:.4f}  spec={v_spec:.4f}")

        if test_loader is not None:
            test_res = evaluate_with_tta(model, test_loader, device, num_classes,
                                         thresholds=thr, class_names=class_names)
            probs_test_by_variant[variant] = test_res["probs"]
            if test_labels_shared is None:
                test_labels_shared = test_res["labels"]
            per_variant_test[variant] = {
                "macro_auc": test_res["macro_auc"], "per_class_auc": test_res["per_class_auc"],
                "macro_f1": test_res["macro_f1"], "kappa": test_res["kappa"],
                "sensitivity": test_res["sensitivity"], "specificity": test_res["specificity"],
            }
            t_sens = float(np.mean(list(test_res["sensitivity"].values())))
            t_spec = float(np.mean(list(test_res["specificity"].values())))
            print(f"  [test] f1={test_res['macro_f1']:.4f}  kappa={test_res['kappa']:.4f}  "
                  f"sens={t_sens:.4f}  spec={t_spec:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Soft routing ensemble ────────────────────────────────────────────
    print(f"\n{'='*60}\nSoft routing (uniform average)...")
    assert val_labels_shared is not None, "No variant produced a valid result."
    prob_soft_val = soft_route_uniform(probs_val_by_variant, class_names)
    thr_soft = find_optimal_thresholds(prob_soft_val, val_labels_shared, num_classes)
    met_soft_val = compute_metrics(val_labels_shared, prob_soft_val, thr_soft, class_names)
    print(f"  [val]  f1={met_soft_val['macro_f1']:.4f}  kappa={met_soft_val['kappa']:.4f}")

    met_soft_test = None
    if test_labels_shared is not None:
        prob_soft_test = soft_route_uniform(probs_test_by_variant, class_names)
        met_soft_test = compute_metrics(test_labels_shared, prob_soft_test, thr_soft, class_names)
        print(f"  [test] f1={met_soft_test['macro_f1']:.4f}  kappa={met_soft_test['kappa']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    combined = {
        "experiment_name": exp_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config_snapshot": cfg,
        "split": "8:1:1",
        "val": {
            "per_variant_metrics": _to_serializable(per_variant_val),
            "soft_routing": _to_serializable({**met_soft_val, "thresholds": thr_soft.tolist()}),
        },
        "test": {
            "per_variant_metrics": _to_serializable(per_variant_test),
            "soft_routing": _to_serializable(met_soft_test),
        } if test_labels_shared is not None else None,
    }
    json_path = os.path.join(output_base, f"{exp_name}.json")
    with open(json_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nResults → {json_path}\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task B — CLAHE + cRT (EfficientNet-B3)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke: 200/50, 2 epochs/stage")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()

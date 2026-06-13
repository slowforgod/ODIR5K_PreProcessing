"""
train/train_v7.py

Entry point for V7 — EfficientNet-B3 전용 레시피 (V6에서 분기).

V6(공용 train_v6.py) 대비 EfficientNet-B3에 맞춘 차이점:
  (2) backbone(features) / head(classifier) 학습률 분리
        backbone_lr=5e-5, head_lr=5e-4  (config train.* 에서 읽음)
  (3) best checkpoint 선택 기준 = val macro F1  (V6와 동일하게 유지)
  (4) H 클래스 집중 보강 — 손실함수 변경 없이 데이터 전처리 단에서:
        (a) LSE per-class target / max_aug 차등화 (H target 1100, max_aug 18)
        (b) 소수 클래스(H) 복제본(aug_idx>=1) 전용 강증강
  (5) routing 고도화 — soft_uniform 에 더해 클래스별 val AUC 가중
        soft_weighted 라우팅을 함께 계산/저장

Usage:
    python -m train_EfficientNet.train_v7 --config configs_EfficientNet/v7_efficientnet_efficientnet.yaml
    python -m train_EfficientNet.train_v7 --config configs_EfficientNet/v7_efficientnet_efficientnet.yaml --smoke
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


def build_param_groups(model, backbone_lr, head_lr):
    """EfficientNet-B3 의 features(backbone) / classifier(head) 를 분리해
    서로 다른 학습률의 param group 을 만든다.

    model.features / model.classifier 가 있으면 둘로 분리하고, 없으면
    (안전망) 단일 그룹으로 fallback 한다. 어느 파라미터에도 중복/누락이
    없도록 id 집합으로 검증한다.
    """
    has_split = hasattr(model, "features") and hasattr(model, "classifier")
    if not has_split:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": head_lr}]

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params     = [p for p in model.classifier.parameters() if p.requires_grad]

    covered = {id(p) for p in backbone_params} | {id(p) for p in head_params}
    leftover = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if leftover:
        # avgpool/mixup 등에는 보통 파라미터가 없지만, 혹시 있으면 head 그룹에 합류.
        head_params = head_params + leftover

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class V1ClaheCache:
    """Variant-scoped cache for (crop_black_border → V1 CLAHE → Resize) result.

    Eliminates redundant V1 CLAHE computation across the 3 loss runs (and
    all epochs) within a single variant. Keyed by filename, so LSE-expanded
    records sharing a source image hit the same cache entry.
    """
    def __init__(self, image_dir, v1_preprocessor, img_size):
        self.image_dir = image_dir
        self.v1 = v1_preprocessor
        self.img_size = img_size
        self.store: Dict[str, np.ndarray] = {}

    def get(self, filename: str, label) -> np.ndarray:
        cached = self.store.get(filename)
        if cached is not None:
            return cached
        img_path = os.path.join(self.image_dir, filename)
        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img_np = crop_black_border(img_np)
        img_np, _ = self.v1.apply(img_np, label)
        img_np = np.array(Image.fromarray(img_np).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        ))
        self.store[filename] = img_np
        return img_np

    def clear(self) -> None:
        self.store.clear()


class ODIRV7TrainDataset(Dataset):
    """Train dataset: cache(crop+V1+Resize) → V2 aug → ToTensor+Normalize.

    (4-b) 소수 클래스 강증강:
        aug_idx >= 1 (LSE 복제본) 이면서 라벨에 strong_aug 대상 클래스가
        하나라도 양성이면 strong_v2(강한 증강)를 적용하고, 그 외에는 기본
        v2 를 적용한다. 원본(aug_idx==0)은 항상 기본 증강만 받는다.
    """
    def __init__(self, records, image_dir, img_size=224, cache=None,
                 v2_preprocessor=None, strong_v2_preprocessor=None,
                 strong_class_indices=None, smoke_n=0):
        if smoke_n > 0:
            records = records[:smoke_n]
        self.cache = cache
        self.v2 = v2_preprocessor
        self.strong_v2 = strong_v2_preprocessor
        self.strong_class_indices = set(strong_class_indices or [])
        self.image_dir = image_dir
        self.img_size = img_size
        self._to_tensor_norm = T.Compose([
            T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self.samples = []
        missing = 0
        n_strong = 0
        for rec in records:
            fname = rec["filename"]
            img_path = os.path.join(image_dir, fname)
            if not os.path.isfile(img_path):
                missing += 1
                continue
            label = rec["labels"].copy()
            aug_idx = int(rec.get("aug_idx", 0))
            use_strong = (
                self.strong_v2 is not None
                and aug_idx >= 1
                and any(label[ci] == 1.0 for ci in self.strong_class_indices)
            )
            if use_strong:
                n_strong += 1
            self.samples.append((fname, label, use_strong))
        if missing:
            print(f"  [warn] {missing} records skipped", file=sys.stderr)
        if self.strong_v2 is not None:
            print(f"  [strong-aug] {n_strong} minority-class copies routed to strong augmentation")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label, use_strong = self.samples[idx]
        img_np = self.cache.get(fname, label)
        aug = self.strong_v2 if use_strong else self.v2
        if aug is not None:
            # V2's Resize is a no-op since cache is already (img_size, img_size).
            img_np, _ = aug.apply(img_np, label)
        return self._to_tensor_norm(img_np), torch.tensor(label, dtype=torch.float32)


class ODIRV7EvalDataset(Dataset):
    """Val/Test dataset: cache(crop+V1+Resize) → ToTensor+Normalize.

    No V2 augmentation. Shares the V1 CLAHE cache with the train dataset.
    """
    def __init__(self, records, image_dir, img_size=224, cache=None, smoke_n=0):
        if smoke_n > 0:
            records = records[:smoke_n]
        self.cache = cache
        self.image_dir = image_dir
        self.img_size = img_size
        self._to_tensor_norm = T.Compose([
            T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self.samples = []
        missing = 0
        for rec in records:
            fname = rec["filename"]
            img_path = os.path.join(image_dir, fname)
            if not os.path.isfile(img_path):
                missing += 1
                continue
            self.samples.append((fname, rec["labels"].copy()))
        if missing:
            print(f"  [warn] {missing} records skipped", file=sys.stderr)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_np = self.cache.get(fname, label)
        return self._to_tensor_norm(img_np), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop — Manifold Mixup with mixup_prob
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, alpha, mixup_layers,
                    scaler=None, mixup_prob=1.0, grad_clip=0.0):
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
                    logits, label_a, label_b, lam = model(imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers)
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
                logits, label_a, label_b, lam = model(imgs, y=labels, alpha=alpha, mixup_layers=mixup_layers)
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
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict, smoke: bool = False) -> None:
    import pandas as pd
    from model.base import build_model
    from preprocessing.v1.clahe import V1Preprocessor
    from preprocessing.v1.routing import (
        DEFAULT_VARIANTS, soft_route_uniform, soft_route_weighted,
    )
    from preprocessing.v2.augmentation import V2Preprocessor
    from preprocessing.v2.oversampling import expand_with_lse
    from preprocessing.v2.losses import effective_number_class_weights, make_criterion
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

    # ── Data loading + splits ─────────────────────────────────────────────
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
    cb_weights    = effective_number_class_weights(labels_ref_np, beta=cb_beta, normalize=cb_normalize)
    print(f"[CB weights from original train_df ({len(train_df)} rows)]")
    for i, c in enumerate(class_cols):
        print(f"  {c}: {cb_weights[i]:.4f}")

    # ── LSE expansion — (4-a) per-class target / max_aug 차등화 ──────────
    pre_cfg      = cfg.get("preprocessing", {})
    target_count = int(pre_cfg.get("lse_custom_target_count", 850))
    max_aug      = int(pre_cfg.get("lse_max_aug_per_source", 12))
    per_class_target  = pre_cfg.get("lse_per_class_target", {}) or {}
    per_class_max_aug = pre_cfg.get("lse_per_class_max_aug", {}) or {}

    expanded_records = expand_with_lse(
        train_df, class_cols=class_cols,
        target_count=target_count, max_aug_per_source=max_aug, seed=seed,
        per_class_target=per_class_target, per_class_max_aug=per_class_max_aug,
    )

    def _df_to_records(df):
        return [{"filename": str(row["filename"]),
                 "labels": row[class_cols].values.astype(np.float32),
                 "aug_idx": 0}
                for _, row in df.iterrows()]

    val_records  = _df_to_records(val_df)
    test_records = _df_to_records(test_df) if test_df is not None else None

    smoke_train = 200 if smoke else 0
    smoke_val   = 50  if smoke else 0
    smoke_test  = 50  if smoke else 0

    # ── Config ───────────────────────────────────────────────────────────
    variants     = pre_cfg.get("variants", DEFAULT_VARIANTS)
    clip_limit   = float(pre_cfg.get("clip_limit", 2.0))
    tile_grid    = tuple(pre_cfg.get("tile_grid_size", [8, 8]))
    alpha        = float(pre_cfg.get("mixup_alpha", 0.2))
    mixup_layers = list(pre_cfg.get("mixup_layers", [0]))
    mixup_prob   = float(pre_cfg.get("mixup_prob", 1.0))

    aug_keys = (
        "hflip_p", "vflip_p", "geo_p", "rotate_limit", "affine_scale",
        "affine_translate", "affine_rotate", "affine_shear", "crop_scale",
        "color_jitter_p", "brightness", "contrast", "saturation", "blur_p", "blur_limit",
    )
    aug_kwargs = {k: pre_cfg[k] for k in aug_keys if k in pre_cfg}

    # (4-b) 소수 클래스 강증강 설정 — 기본 aug_kwargs 위에 strong_aug 로 덮어씀.
    strong_aug_classes  = pre_cfg.get("strong_aug_classes", []) or []
    strong_aug_cfg      = pre_cfg.get("strong_aug", {}) or {}
    strong_class_idx    = [class_cols.index(c) for c in strong_aug_classes if c in class_cols]
    use_strong_aug      = bool(strong_class_idx) and bool(strong_aug_cfg)
    strong_aug_kwargs   = {**aug_kwargs, **{k: v for k, v in strong_aug_cfg.items() if k in aug_keys}}
    if use_strong_aug:
        print(f"[strong-aug] enabled for classes {strong_aug_classes} "
              f"(indices {strong_class_idx})")

    train_cfg    = cfg["train"]
    batch_size   = int(train_cfg.get("batch_size", 32))
    num_epochs   = 2 if smoke else int(train_cfg.get("num_epochs", 30))
    patience     = int(train_cfg.get("patience", 6))
    lr           = float(train_cfg.get("lr", 1e-4))
    # (2) backbone / head 분리 학습률 (없으면 단일 lr 로 fallback)
    backbone_lr  = float(train_cfg.get("backbone_lr", lr))
    head_lr      = float(train_cfg.get("head_lr", lr))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    dropout      = float(train_cfg.get("dropout", 0.4))
    num_workers  = 0 if smoke else int(train_cfg.get("num_workers", 4))
    grad_clip    = float(train_cfg.get("grad_clip", 0.0))
    eta_min      = float(train_cfg.get("eta_min", 0.0))
    print(f"[LR] backbone_lr={backbone_lr:g}  head_lr={head_lr:g}  weight_decay={weight_decay:g}")

    loss_experiments = cfg.get("losses", {}).get("experiments",
        ["robust_asymmetric_loss", "cb_bce_with_logits_loss", "cb_focal_loss"])
    loss_params = {k: cfg.get("losses", {}).get(k) for k in cfg.get("losses", {})}

    output_base = cfg.get("output_dir", "analysis/EfficientNet/v7_analysis")
    exp_name    = cfg.get("experiment_name", "v7_efficientnet_seed42")
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
        v2_pre = V2Preprocessor(img_size=img_size, **aug_kwargs)
        strong_v2_pre = (
            V2Preprocessor(img_size=img_size, **strong_aug_kwargs)
            if use_strong_aug else None
        )
        # Shared V1 CLAHE cache for this variant: reused across train/val/test
        # datasets and across all 3 loss runs × num_epochs.
        clahe_cache = V1ClaheCache(image_dir, v1_pre, img_size)

        train_ds = ODIRV7TrainDataset(expanded_records, image_dir, img_size,
                                      cache=clahe_cache, v2_preprocessor=v2_pre,
                                      strong_v2_preprocessor=strong_v2_pre,
                                      strong_class_indices=strong_class_idx,
                                      smoke_n=smoke_train)
        val_ds   = ODIRV7EvalDataset(val_records, image_dir, img_size,
                                     cache=clahe_cache, smoke_n=smoke_val)
        test_ds  = (
            ODIRV7EvalDataset(test_records, image_dir, img_size,
                              cache=clahe_cache, smoke_n=smoke_test)
            if test_records is not None else None
        )

        print(f"  Train (expanded): {len(train_ds)}  Val: {len(val_ds)}"
              + (f"  Test: {len(test_ds)}" if test_ds else ""))

        # persistent_workers keeps worker processes alive across epoch
        # boundaries so each worker's V1 CLAHE cache survives between
        # epochs AND between the 3 loss runs (loaders are variant-scoped).
        loader_persist = num_workers > 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  persistent_workers=loader_persist,
                                  pin_memory=(device.type == "cuda"))
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers,
                                  persistent_workers=loader_persist,
                                  pin_memory=(device.type == "cuda"))
        test_loader  = (
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers,
                       persistent_workers=loader_persist,
                       pin_memory=(device.type == "cuda"))
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
            # (2) backbone / head 학습률 분리 param group
            optimizer = torch.optim.Adam(
                build_param_groups(model, backbone_lr, head_lr),
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
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
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    alpha, mixup_layers, scaler, mixup_prob, grad_clip,
                )

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
                val_auc   = epoch_metrics["macro_auc"]
                val_f1    = epoch_metrics["macro_f1"]
                val_kappa = epoch_metrics["kappa"]
                val_sens  = float(np.mean(list(epoch_metrics["sensitivity"].values())))
                val_spec  = float(np.mean(list(epoch_metrics["specificity"].values())))
                scheduler.step()

                print(f"    Epoch {epoch:03d}/{num_epochs} | loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | "
                      f"val_kappa={val_kappa:.4f} | val_sens={val_sens:.4f} | val_spec={val_spec:.4f}")

                with open(history_path, "a", newline="") as hf:
                    writer = csv.DictWriter(hf, fieldnames=history_fields)
                    writer.writerow({
                        "epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6), "val_macro_auc": round(val_auc, 6),
                        "val_macro_f1": round(val_f1, 6), "val_kappa": round(val_kappa, 6),
                    })

                # (3) best checkpoint 선택 기준 = val macro F1
                if not np.isnan(val_f1) and val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch = epoch
                    no_improve = 0
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"      [saved] best.pth (val_f1={best_f1:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"    [early stop] no improvement for {patience} epochs")
                        break

            # ── Post-training: val 평가 + threshold 최적화 ──────────────
            if best_f1 >= 0 and os.path.isfile(ckpt_path):
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

                # test 평가 — val threshold 재사용
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
                probs_val_by_loss_by_variant[loss_name][variant] = np.zeros((len(val_ds), num_classes), dtype=np.float32)
                per_variant_per_loss_val[variant][loss_name] = {
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

        # ── End of variant: free V1 CLAHE cache + tear down loaders ─────
        print(f"  [cache] V1 CLAHE cache hit {len(clahe_cache.store)} unique images "
              f"for variant {variant}; clearing.")
        clahe_cache.clear()
        del train_loader, val_loader
        if test_loader is not None:
            del test_loader

    # ── Routing aggregation — (5) soft_uniform + soft_weighted ───────────
    print(f"\n{'='*60}")
    print("Computing routing metrics per loss (soft_uniform + soft_weighted)...")
    assert val_labels_shared is not None, "No run produced a valid eval result."

    val_routing_metrics:  Dict[str, dict] = {}
    test_routing_metrics: Dict[str, dict] = {}

    for loss_name in loss_experiments:
        probs_val_variant = probs_val_by_loss_by_variant[loss_name]

        # per-variant per-class val AUC 를 soft_weighted 가중치로 사용.
        per_variant_class_weight = {
            variant: per_variant_per_loss_val[variant][loss_name].get("per_class_auc", {}) or {}
            for variant in probs_val_variant
        }

        val_routing_metrics[loss_name] = {}
        test_routing_metrics[loss_name] = {}

        # --- soft_uniform ---
        prob_soft_val = soft_route_uniform(probs_val_variant, class_names)
        thr_soft      = find_optimal_thresholds(prob_soft_val, val_labels_shared, num_classes)
        met_soft_val  = compute_metrics(val_labels_shared, prob_soft_val, thr_soft, class_names)
        val_routing_metrics[loss_name]["soft_uniform"] = {**met_soft_val, "thresholds": thr_soft.tolist()}
        print(f"  [val]  {loss_name:<28} uniform  auc={met_soft_val['macro_auc']:.4f}  f1={met_soft_val['macro_f1']:.4f}")

        # --- soft_weighted (per-class val AUC) ---
        prob_w_val = soft_route_weighted(probs_val_variant, per_variant_class_weight, class_names)
        thr_w      = find_optimal_thresholds(prob_w_val, val_labels_shared, num_classes)
        met_w_val  = compute_metrics(val_labels_shared, prob_w_val, thr_w, class_names)
        val_routing_metrics[loss_name]["soft_weighted"] = {**met_w_val, "thresholds": thr_w.tolist()}
        print(f"  [val]  {loss_name:<28} weighted auc={met_w_val['macro_auc']:.4f}  f1={met_w_val['macro_f1']:.4f}")

        if test_labels_shared is not None:
            probs_test_variant = probs_test_by_loss_by_variant[loss_name]

            prob_soft_test = soft_route_uniform(probs_test_variant, class_names)
            met_soft_test  = compute_metrics(test_labels_shared, prob_soft_test, thr_soft, class_names)
            test_routing_metrics[loss_name]["soft_uniform"] = met_soft_test

            prob_w_test = soft_route_weighted(probs_test_variant, per_variant_class_weight, class_names)
            met_w_test  = compute_metrics(test_labels_shared, prob_w_test, thr_w, class_names)
            test_routing_metrics[loss_name]["soft_weighted"] = met_w_test
            print(f"  [test] {loss_name:<28} uniform  auc={met_soft_test['macro_auc']:.4f}  "
                  f"weighted auc={met_w_test['macro_auc']:.4f}")

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
    parser = argparse.ArgumentParser(
        description="Train V7: EfficientNet-B3 (split LR + H-focused aug + weighted routing)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 200 train / 50 val, 2 epochs per run")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()

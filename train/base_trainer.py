"""
train/base_trainer.py

Common training loop for V0, V1, V2, V4.
MixupTrainer (V3, V5, V6) inherits from this class.

변경 이력:
  - split JSON: patient_split_7class_stratified.json (8:2)
              → patient_split_7class_stratified_811.json (8:1:1)
  - _setup(): test split 로드 + test_loader 추가
  - evaluate_and_save(): test 평가 추가, 결과를 val/test 키로 분리 저장
"""

import csv
import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from model.base import build_model
from analysis.metrics import evaluate_with_tta, save_metrics


# ---------------------------------------------------------------------------
# Allowed config values
# ---------------------------------------------------------------------------

ALLOWED_SELECTION = {"macro_auc", "macro_f1", "macro_kappa"}
ALLOWED_LOSS = {"weighted_bce", "focal"}


# ---------------------------------------------------------------------------
# Focal Loss (logits)
# ---------------------------------------------------------------------------

class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        p = torch.sigmoid(logits)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        focal_w = (1.0 - p_t) ** self.gamma
        return (focal_w * bce).mean()


def _build_criterion(
    loss_type: str,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    focal_use_pos_weight: bool,
    device: torch.device,
) -> nn.Module:
    loss_type = loss_type.lower().strip()
    if loss_type not in ALLOWED_LOSS:
        raise ValueError(
            f"loss_type must be one of {sorted(ALLOWED_LOSS)}, got '{loss_type}'"
        )
    if loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    pw = pos_weight if focal_use_pos_weight else None
    return FocalLossWithLogits(gamma=focal_gamma, pos_weight=pw).to(device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ODIRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        class_cols: list,
        filename_col: str = "filename",
        transform=None,
        img_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.class_cols = class_cols
        self.filename_col = filename_col
        self.transform = transform
        self.img_size = img_size

        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row[self.filename_col])
        image = np.array(Image.open(img_path).convert("RGB"))
        label = row[self.class_cols].values.astype(np.float32)

        if self.transform is not None:
            image, label = self.transform(image, label)

        pil = Image.fromarray(image).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        tensor = self._to_tensor(pil)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_pos_weight(train_df: pd.DataFrame, class_cols: list, device):
    pos = train_df[class_cols].sum(axis=0).values.astype(np.float32)
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)
    return pos_weight.to(device)


def _build_sampler(train_df: pd.DataFrame, class_cols: list):
    labels = train_df[class_cols].values.astype(np.float32)
    class_freq = labels.mean(axis=0)
    inv_freq = 1.0 / np.clip(class_freq, 1e-6, None)
    sample_weights = (labels * inv_freq[None, :]).sum(axis=1)
    sample_weights = np.clip(sample_weights, 1e-6, None)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------

class BaseTrainer:
    """Training loop for V0 / V1 / V2 / V4."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        cfg = self.cfg
        train_cfg = cfg["train"]
        data_cfg = cfg["data"]
        prep_cfg = cfg.get("preprocessing", {})

        _set_seed(train_cfg["seed"])

        self.device = torch.device(
            train_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Split 로드 ──────────────────────────────────────────────────
        df = pd.read_csv(data_cfg["csv_path"])
        with open(data_cfg["split_path"]) as f:
            split = json.load(f)

        train_ids = set(split[data_cfg["split_train_key"]])
        val_ids   = set(split[data_cfg["split_val_key"]])
        # ★ test_patients — config에 키가 있을 때만 로드 (하위 호환)
        test_key  = data_cfg.get("split_test_key")
        test_ids  = set(split[test_key]) if test_key and test_key in split else None

        self.train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)]
        self.val_df   = df[df[data_cfg["patient_id_col"]].isin(val_ids)]
        self.test_df  = (
            df[df[data_cfg["patient_id_col"]].isin(test_ids)]
            if test_ids is not None else None
        )

        self.class_cols  = data_cfg["class_cols"]
        self.class_names = cfg["class_names"]
        self.img_size    = data_cfg.get("img_size", 224)

        # ── Preprocessor & Transforms ───────────────────────────────────
        preprocessor   = self._build_preprocessor(prep_cfg)
        train_transform = preprocessor.apply
        val_transform   = None if preprocessor.is_train_only() else preprocessor.apply

        # ── Datasets ────────────────────────────────────────────────────
        train_ds = ODIRDataset(
            self.train_df, data_cfg["image_dir"], self.class_cols,
            data_cfg["filename_col"], train_transform, self.img_size,
        )
        val_ds = ODIRDataset(
            self.val_df, data_cfg["image_dir"], self.class_cols,
            data_cfg["filename_col"], val_transform, self.img_size,
        )

        # ── DataLoaders ─────────────────────────────────────────────────
        sampler     = _build_sampler(self.train_df, self.class_cols)
        num_workers = train_cfg.get("num_workers", 4)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg["batch_size"],
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        # ★ test_loader — test_df가 있을 때만 생성
        if self.test_df is not None:
            test_ds = ODIRDataset(
                self.test_df, data_cfg["image_dir"], self.class_cols,
                data_cfg["filename_col"], val_transform, self.img_size,
            )
            self.test_loader = DataLoader(
                test_ds,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            self.test_loader = None

        # ── Model ────────────────────────────────────────────────────────
        self.model = build_model(
            name=cfg["model_name"],
            num_classes=cfg["num_classes"],
            pretrained=True,
            dropout=train_cfg.get("dropout", 0.5),
        ).to(self.device)

        # ── Loss ─────────────────────────────────────────────────────────
        pos_weight         = _build_pos_weight(self.train_df, self.class_cols, self.device)
        loss_type          = str(train_cfg.get("loss_type", "weighted_bce"))
        focal_gamma        = float(train_cfg.get("focal_gamma", 2.0))
        focal_use_pw       = bool(train_cfg.get("focal_use_pos_weight", True))
        self.criterion     = _build_criterion(
            loss_type, pos_weight, focal_gamma, focal_use_pw, self.device,
        )
        self.loss_type = loss_type

        # ── Optimizer / Scheduler ────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["num_epochs"]
        )

        # ── Training state ───────────────────────────────────────────────
        self.num_epochs      = train_cfg["num_epochs"]
        self.patience        = train_cfg.get("patience", 8)
        self.output_dir      = cfg["output_dir"]
        self.experiment_name = cfg["experiment_name"]
        os.makedirs(self.output_dir, exist_ok=True)

        run_tag = train_cfg.get("run_tag")
        suffix  = f"_{run_tag}" if run_tag else ""
        self.run_tag            = run_tag
        self._ckpt_filename     = f"best{suffix}.pth"
        self._thr_filename      = f"thresholds{suffix}.npy"
        self._history_filename  = f"history{suffix}.csv"

        selection_metric = str(train_cfg.get("selection_metric", "macro_auc")).lower()
        if selection_metric not in ALLOWED_SELECTION:
            raise ValueError(
                f"selection_metric must be one of {sorted(ALLOWED_SELECTION)}, "
                f"got '{selection_metric}'"
            )
        self.selection_metric = selection_metric
        self._selection_key   = (
            "kappa" if selection_metric == "macro_kappa" else selection_metric
        )

        self.best_score      = -1.0
        self.patience_counter = 0
        self.best_thresholds: Optional[np.ndarray] = None

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v0.preprocess import V0Preprocessor
        return V0Preprocessor(**prep_cfg)

    # ------------------------------------------------------------------
    # Train / Val steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for imgs, labels in self.train_loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(imgs)
                loss   = self.criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _val_step(self) -> dict:
        result = evaluate_with_tta(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=len(self.class_cols),
            thresholds=None,          # val에서 threshold 최적화
            class_names=self.class_names,
        )
        return result

    # ------------------------------------------------------------------
    # History CSV
    # ------------------------------------------------------------------

    def _append_history(self, epoch: int, train_loss: float, val_result: dict):
        path = os.path.join(self.output_dir, self._history_filename)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "train_loss", "val_macro_auc",
                    "val_macro_f1", "val_kappa",
                ])
            writer.writerow([
                epoch,
                round(train_loss, 6),
                round(val_result.get("macro_auc", float("nan")), 6),
                round(val_result.get("macro_f1",  float("nan")), 6),
                round(val_result.get("kappa",     float("nan")), 6),
            ])

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self):
        print(f"[{self.experiment_name}] Training on {self.device}")
        print(f"  Train : {len(self.train_loader.dataset)} samples")
        print(f"  Val   : {len(self.val_loader.dataset)} samples")
        if self.test_loader is not None:
            print(f"  Test  : {len(self.test_loader.dataset)} samples")
        print(f"  Epochs: {self.num_epochs}  Patience: {self.patience}")
        print(f"  Loss: {self.loss_type}  |  Selection: val_{self.selection_metric}\n")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch()
            val_result = self._val_step()

            self._append_history(epoch, train_loss, val_result)
            self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_auc={val_result.get('macro_auc', -1):.4f} | "
                f"val_f1={val_result.get('macro_f1', 0):.4f} | "
                f"val_kappa={val_result.get('kappa', 0):.4f}"
            )

            current = val_result.get(self._selection_key, float("nan"))
            if not np.isnan(current) and current > self.best_score:
                self.best_score      = current
                self.best_thresholds = val_result["thresholds"]
                self.patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, self._ckpt_filename),
                )
                np.save(
                    os.path.join(self.output_dir, self._thr_filename),
                    self.best_thresholds,
                )
                print(f"  ★ Best saved (val_{self.selection_metric}={self.best_score:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\nTraining done. Best val_{self.selection_metric}: {self.best_score:.4f}")

    # ------------------------------------------------------------------
    # evaluate_and_save  ★ 핵심 변경
    # ------------------------------------------------------------------

    def evaluate_and_save(self):
        """
        1. best checkpoint 로드
        2. val 재평가  — best threshold 재사용 (재계산 금지)
        3. test 평가   — 동일 threshold 재사용 (재계산 금지)
        4. val/test 분리 구조로 JSON 저장
        """
        # best checkpoint 로드
        ckpt_path = os.path.join(self.output_dir, self._ckpt_filename)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        # best threshold 로드
        thresholds = np.load(os.path.join(self.output_dir, self._thr_filename))

        # ── val 재평가 ────────────────────────────────────────────────
        val_result = evaluate_with_tta(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=len(self.class_cols),
            thresholds=thresholds,      # 재계산 금지
            class_names=self.class_names,
        )
        val_result.pop("probs", None)
        val_result.pop("labels", None)
        val_result.pop("thresholds", None)

        # ── test 평가 ─────────────────────────────────────────────────
        test_result = None
        if self.test_loader is not None:
            test_result = evaluate_with_tta(
                model=self.model,
                loader=self.test_loader,
                device=self.device,
                num_classes=len(self.class_cols),
                thresholds=thresholds,  # val threshold 그대로 재사용
                class_names=self.class_names,
            )
            test_result.pop("probs", None)
            test_result.pop("labels", None)
            test_result.pop("thresholds", None)
            print(
                f"Test  | auc={test_result.get('macro_auc', -1):.4f} "
                f"f1={test_result.get('macro_f1', 0):.4f} "
                f"kappa={test_result.get('kappa', 0):.4f}"
            )
        else:
            print("Warning: test_loader 없음 — test 평가 생략")

        # ── 결과 조합 및 저장 ─────────────────────────────────────────
        result = {
            "val": val_result,
            "test": test_result,    # test_loader 없으면 None
        }
        json_path = save_metrics(
            metrics=result,
            save_dir=self.output_dir,
            experiment_name=self.experiment_name,
            config_snapshot=self.cfg,
        )
        print(f"Results saved → {json_path}")
        return result
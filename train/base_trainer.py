"""
train/base_trainer.py

Common training loop for V0, V1, V2, V4.
MixupTrainer (V3, V5, V6) inherits from this class.
"""

import csv
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
    """Multi-label focal loss operating on raw logits.

    Loss per element = (1 - p_t)^gamma * BCEWithLogits(logits, target)
    where p_t = sigmoid(logits) if target == 1 else 1 - sigmoid(logits).

    Optional pos_weight is multiplied into the underlying BCE term, matching
    nn.BCEWithLogitsLoss(pos_weight=...) semantics. Useful for boosting
    minority-class sensitivity (A / H / M classes in ODIR-5K).

    Args:
        gamma      : focusing parameter (higher → more focus on hard examples)
        pos_weight : optional (num_classes,) tensor on the same device as logits
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        if pos_weight is not None:
            # buffer follows .to(device) on the parent nn.Module
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
    """Build the loss criterion based on config.

    Args:
        loss_type            : "weighted_bce" | "focal"
        pos_weight           : (num_classes,) tensor on `device`
        focal_gamma          : gamma for focal loss
        focal_use_pos_weight : whether focal loss also uses pos_weight
        device               : target device for criterion buffers

    Returns:
        nn.Module ready to call as criterion(logits, target).
    """
    loss_type = loss_type.lower().strip()
    if loss_type not in ALLOWED_LOSS:
        raise ValueError(
            f"loss_type must be one of {sorted(ALLOWED_LOSS)}, got '{loss_type}'"
        )
    if loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # focal
    pw = pos_weight if focal_use_pos_weight else None
    return FocalLossWithLogits(gamma=focal_gamma, pos_weight=pw).to(device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ODIRDataset(Dataset):
    """ODIR-5K multi-label dataset.

    Args:
        df          : filtered DataFrame (train or val split)
        image_dir   : path to preprocessed_images/
        class_cols  : list of 7 class column names
        filename_col: column name for image filename
        transform   : callable(image_np) → image_np  (preprocessor.apply)
        img_size    : resize target (default 224)
    """

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

        # Load as RGB numpy array (H, W, 3) uint8
        image = np.array(Image.open(img_path).convert("RGB"))
        label = row[self.class_cols].values.astype(np.float32)

        # Apply preprocessor (CLAHE, augmentation, etc.)
        if self.transform is not None:
            image, label = self.transform(image, label)

        # Resize → Tensor → Normalize
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


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _build_pos_weight(train_df: pd.DataFrame, class_cols: list, device):
    """pos_weight[i] = (num_negative) / (num_positive) for BCEWithLogitsLoss."""
    pos = train_df[class_cols].sum(axis=0).values.astype(np.float32)
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / np.clip(pos, 1, None), dtype=torch.float32)
    return pos_weight.to(device)


def _build_sampler(train_df: pd.DataFrame, class_cols: list):
    """WeightedRandomSampler to up-sample rare classes."""
    labels = train_df[class_cols].values.astype(np.float32)  # (N, 7)
    class_freq = labels.mean(axis=0)                          # (7,)
    inv_freq = 1.0 / np.clip(class_freq, 1e-6, None)         # (7,)
    sample_weights = (labels * inv_freq[None, :]).sum(axis=1) # (N,)
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
    """Training loop for V0 / V1 / V2 / V4.

    Usage:
        trainer = BaseTrainer(cfg)
        trainer.fit()
        trainer.evaluate_and_save()
    """

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

        # Seed
        _set_seed(train_cfg["seed"])

        # Device
        self.device = torch.device(
            train_cfg["device"]
            if torch.cuda.is_available()
            else "cpu"
        )

        # AMP — only on CUDA
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Data
        df = pd.read_csv(data_cfg["csv_path"])
        import json
        with open(data_cfg["split_path"]) as f:
            split = json.load(f)

        train_ids = set(split[data_cfg["split_train_key"]])
        val_ids = set(split[data_cfg["split_val_key"]])

        self.train_df = df[df[data_cfg["patient_id_col"]].isin(train_ids)]
        self.val_df = df[df[data_cfg["patient_id_col"]].isin(val_ids)]
        self.class_cols = data_cfg["class_cols"]
        self.class_names = cfg["class_names"]
        self.img_size = data_cfg.get("img_size", 224)

        # Preprocessor
        preprocessor = self._build_preprocessor(prep_cfg)
        train_transform = preprocessor.apply
        val_transform = None if preprocessor.is_train_only() else preprocessor.apply

        # Datasets
        train_ds = ODIRDataset(
            self.train_df, data_cfg["image_dir"], self.class_cols,
            data_cfg["filename_col"], train_transform, self.img_size,
        )
        val_ds = ODIRDataset(
            self.val_df, data_cfg["image_dir"], self.class_cols,
            data_cfg["filename_col"], val_transform, self.img_size,
        )

        # Sampler + DataLoaders
        sampler = _build_sampler(self.train_df, self.class_cols)
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

        # Model
        self.model = build_model(
            name=cfg["model_name"],
            num_classes=cfg["num_classes"],
            pretrained=True,
            dropout=train_cfg.get("dropout", 0.5),
        ).to(self.device)

        # Loss — selectable via train.loss_type (default: weighted_bce)
        pos_weight = _build_pos_weight(self.train_df, self.class_cols, self.device)
        loss_type = str(train_cfg.get("loss_type", "weighted_bce"))
        focal_gamma = float(train_cfg.get("focal_gamma", 2.0))
        focal_use_pw = bool(train_cfg.get("focal_use_pos_weight", True))
        self.criterion = _build_criterion(
            loss_type, pos_weight, focal_gamma, focal_use_pw, self.device,
        )
        self.loss_type = loss_type

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["num_epochs"]
        )

        # Training state
        self.num_epochs = train_cfg["num_epochs"]
        self.patience = train_cfg.get("patience", 8)
        self.output_dir = cfg["output_dir"]
        self.experiment_name = cfg["experiment_name"]
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- Run tag (optional file suffix) ----
        # Allows multiple runs in the same output_dir (e.g. weighted_bce vs focal)
        # to coexist without overwriting best/thresholds/history.
        # If unset → keeps the original filenames (best.pth …).
        run_tag = train_cfg.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        self.run_tag = run_tag
        self._ckpt_filename = f"best{suffix}.pth"
        self._thr_filename = f"thresholds{suffix}.npy"
        self._history_filename = f"history{suffix}.csv"

        # ---- Best model selection metric ----
        # Default "macro_auc" preserves existing behaviour for V0/V2/V4.
        # V1 overrides via configs/v1.yaml → "macro_f1" or "macro_kappa".
        selection_metric = str(train_cfg.get("selection_metric", "macro_auc")).lower()
        if selection_metric not in ALLOWED_SELECTION:
            raise ValueError(
                f"selection_metric must be one of {sorted(ALLOWED_SELECTION)}, "
                f"got '{selection_metric}'"
            )
        self.selection_metric = selection_metric
        # Map to the key returned by analysis.metrics.compute_metrics.
        # "macro_kappa" is reported as "kappa" inside that dict.
        self._selection_key = (
            "kappa" if selection_metric == "macro_kappa" else selection_metric
        )

        self.best_score = -1.0
        self.patience_counter = 0
        self.best_thresholds: Optional[np.ndarray] = None

    def _build_preprocessor(self, prep_cfg: dict):
        """Override in subclasses to return the appropriate preprocessor."""
        from preprocessing.v0.preprocess import V0Preprocessor
        return V0Preprocessor(**prep_cfg)

    # ------------------------------------------------------------------
    # Train / Val steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _val_step(self) -> dict:
        """Evaluate with TTA; returns metrics dict."""
        result = evaluate_with_tta(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=len(self.class_cols),
            thresholds=None,        # find optimal thresholds this epoch
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
                round(val_result.get("macro_f1", float("nan")), 6),
                round(val_result.get("kappa", float("nan")), 6),
            ])

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self):
        print(f"[{self.experiment_name}] Training on {self.device}")
        print(f"  Train: {len(self.train_loader.dataset)} samples")
        print(f"  Val  : {len(self.val_loader.dataset)} samples")
        print(f"  Epochs: {self.num_epochs}  Patience: {self.patience}")
        print(f"  Loss: {self.loss_type}  |  Selection: val_{self.selection_metric} (maximise)\n")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch()
            val_result = self._val_step()
            val_auc = val_result.get("macro_auc", -1.0)
            val_f1 = val_result.get("macro_f1", 0.0)
            val_kappa = val_result.get("kappa", 0.0)

            self._append_history(epoch, train_loss, val_result)
            self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"val_kappa={val_kappa:.4f}"
            )

            # Best model tracking — by configured selection_metric (maximise)
            current = val_result.get(self._selection_key, float("nan"))
            if not np.isnan(current) and current > self.best_score:
                self.best_score = current
                self.best_thresholds = val_result["thresholds"]
                self.patience_counter = 0
                ckpt_path = os.path.join(self.output_dir, self._ckpt_filename)
                torch.save(self.model.state_dict(), ckpt_path)
                thr_path = os.path.join(self.output_dir, self._thr_filename)
                np.save(thr_path, self.best_thresholds)
                print(f"  ★ Best saved (val_{self.selection_metric}={self.best_score:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\nTraining done. Best val_{self.selection_metric}: {self.best_score:.4f}")

    # ------------------------------------------------------------------
    # evaluate_and_save
    # ------------------------------------------------------------------

    def evaluate_and_save(self):
        """Load best checkpoint, evaluate with saved thresholds, save JSON."""
        ckpt_path = os.path.join(self.output_dir, self._ckpt_filename)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )

        thr_path = os.path.join(self.output_dir, self._thr_filename)
        thresholds = np.load(thr_path)

        result = evaluate_with_tta(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=len(self.class_cols),
            thresholds=thresholds,   # reuse — do NOT recompute
            class_names=self.class_names,
        )

        # Remove large arrays before saving JSON
        result.pop("probs", None)
        result.pop("labels", None)
        result.pop("thresholds", None)

        json_path = save_metrics(
            metrics=result,
            save_dir=self.output_dir,
            experiment_name=self.experiment_name,
            config_snapshot=self.cfg,
        )
        print(f"Results saved → {json_path}")
        return result
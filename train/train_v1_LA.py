"""
train/train_v1_LA.py

Entry point for V1 CLAHE L+A experiment.

Usage:
    python -m train.train_v1_LA --config configs/v1_LA.yaml
"""

import argparse
import yaml

from train.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1LATrainer(BaseTrainer):
    """V1 — CLAHE on L and A channels of LAB colour space."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe_LA import V1LAPreprocessor
        return V1LAPreprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE L+A")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_LA.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1LATrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
"""
train/train_v1_LAB.py

Entry point for V1 CLAHE L+A+B experiment.

Usage:
    python -m train.train_v1_LAB --config configs/v1_LAB.yaml
"""

import argparse
import yaml

from train.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1LABTrainer(BaseTrainer):
    """V1 — CLAHE on L, A, and B channels of LAB colour space."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe_LAB import V1LABPreprocessor
        return V1LABPreprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE L+A+B")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_LAB.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1LABTrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
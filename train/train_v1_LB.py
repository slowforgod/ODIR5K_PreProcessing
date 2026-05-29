"""
train/train_v1_LB.py

Entry point for V1 CLAHE L+B experiment.

Usage:
    python -m train.train_v1_LB --config configs/v1_LB.yaml
"""

import argparse
import yaml

from train.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1LBTrainer(BaseTrainer):
    """V1 — CLAHE on L and B channels of LAB colour space."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe_LB import V1LBPreprocessor
        return V1LBPreprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE L+B")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_LB.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1LBTrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
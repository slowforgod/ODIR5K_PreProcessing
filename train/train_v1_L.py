"""
train/train_v1_L.py

Entry point for V1 CLAHE L-only experiment.

Usage:
    python -m train.train_v1_L --config configs/v1_L.yaml
"""

import argparse
import yaml

from train.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1LTrainer(BaseTrainer):
    """V1 — CLAHE on L channel only."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe import V1Preprocessor
        return V1Preprocessor(variant="V1_L", **prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE L")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_L.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1LTrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
"""
train/train_v1_A.py

Entry point for V1 CLAHE A only experiment.

Usage:
    python -m train.train_v1_A --config configs/v1_A.yaml
"""

import argparse
import yaml

from train.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1ATrainer(BaseTrainer):
    """V1 — CLAHE on A channel only of LAB colour space."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe_A import V1APreprocessor
        return V1APreprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE A only")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1_A.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1ATrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
"""
train/train_v1_B.py

Entry point for V1 CLAHE B only experiment.

Usage:
    python -m train_ResNet.train_v1_B --config configs_ResNet/v1_B.yaml
"""

import argparse
import yaml

from train_ResNet.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V1BTrainer(BaseTrainer):
    """V1 — CLAHE on B channel only of LAB colour space."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v1.clahe_B import V1BPreprocessor
        return V1BPreprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V1 CLAHE B only")
    parser.add_argument(
        "--config",
        type=str,
        default="configs_ResNet/v1_B.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V1BTrainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()
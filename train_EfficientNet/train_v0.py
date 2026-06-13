"""
train/train_v0.py

Entry point for V0 baseline experiment.

Usage:
    python -m train_EfficientNet.train_v0 --config configs_EfficientNet/v0_efficientnet.yaml
"""

import argparse
import yaml

from train_EfficientNet.base_trainer import BaseTrainer


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class V0Trainer(BaseTrainer):
    """V0 Baseline — identity preprocessor (no transform)."""

    def _build_preprocessor(self, prep_cfg: dict):
        from preprocessing.v0.preprocess import V0Preprocessor
        return V0Preprocessor(**prep_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train V0 Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs_EfficientNet/v0_efficientnet.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {cfg['experiment_name']}\n")

    trainer = V0Trainer(cfg)
    trainer.fit()
    trainer.evaluate_and_save()


if __name__ == "__main__":
    main()

"""
Minimal smoke test: loads config, runs 1 variant x 1 loss x 1 epoch,
checks training loop completes and CSV is written.
"""
import sys
import os
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings("ignore")
import yaml
import json
import random
import numpy as np
import torch

def minimal_cfg(version):
    with open(f'configs_ResNet/v{version}.yaml') as f:
        cfg = yaml.safe_load(f)
    # Restrict to 1 variant and 1 loss
    cfg['preprocessing']['variants'] = ['V0']
    cfg['losses']['experiments'] = ['bce_with_logits_loss']
    return cfg

def run_v4_minimal():
    from train_ResNet.train_v4 import train
    cfg = minimal_cfg(4)
    train(cfg, smoke=True)
    # Check output
    json_path = os.path.join('analysis/v4_analysis', 'v4_resnet50_seed42.json')
    assert os.path.isfile(json_path), f"JSON not found: {json_path}"
    with open(json_path) as f:
        out = json.load(f)
    assert 'per_variant_per_loss_metrics' in out, "Missing per_variant_per_loss_metrics"
    assert 'routing_metrics' in out, "Missing routing_metrics"
    assert 'V0' in out['per_variant_per_loss_metrics']
    assert 'bce_with_logits_loss' in out['per_variant_per_loss_metrics']['V0']
    assert 'bce_with_logits_loss' in out['routing_metrics']
    assert 'soft_uniform' in out['routing_metrics']['bce_with_logits_loss']
    print("=== V4 MINIMAL SMOKE PASSED ===", flush=True)

def run_v5_minimal():
    from train_ResNet.train_v5 import train
    cfg = minimal_cfg(5)
    train(cfg, smoke=True)
    json_path = os.path.join('analysis/v5_analysis', 'v5_resnet50_seed42.json')
    assert os.path.isfile(json_path), f"JSON not found: {json_path}"
    with open(json_path) as f:
        out = json.load(f)
    assert 'per_variant_per_loss_metrics' in out
    assert 'routing_metrics' in out
    print("=== V5 MINIMAL SMOKE PASSED ===", flush=True)

def run_v6_minimal():
    from train_ResNet.train_v6 import train
    cfg = minimal_cfg(6)
    train(cfg, smoke=True)
    json_path = os.path.join('analysis/v6_analysis', 'v6_resnet50_seed42.json')
    assert os.path.isfile(json_path), f"JSON not found: {json_path}"
    with open(json_path) as f:
        out = json.load(f)
    assert 'per_variant_per_loss_metrics' in out
    assert 'routing_metrics' in out
    print("=== V6 MINIMAL SMOKE PASSED ===", flush=True)

if __name__ == "__main__":
    v = int(sys.argv[1])
    if v == 4:
        run_v4_minimal()
    elif v == 5:
        run_v5_minimal()
    elif v == 6:
        run_v6_minimal()

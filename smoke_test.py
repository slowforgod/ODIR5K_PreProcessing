"""Quick smoke test runner for V4/V5/V6."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings("ignore")
import yaml

def run(version):
    with open(f'configs_ResNet/v{version}.yaml') as f:
        cfg = yaml.safe_load(f)
    if version == 4:
        from train_ResNet.train_v4 import train
    elif version == 5:
        from train_ResNet.train_v5 import train
    elif version == 6:
        from train_ResNet.train_v6 import train
    print(f"\n=== Smoke V{version} ===", flush=True)
    train(cfg, smoke=True)
    print(f"=== V{version} OK ===", flush=True)

if __name__ == "__main__":
    v = int(sys.argv[1])
    run(v)

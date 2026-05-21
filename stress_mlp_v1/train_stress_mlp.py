from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STRESS_ONLY_ROOT = REPO_ROOT / "stress_only_v1"
for path in (REPO_ROOT / "stress_mlp_v1", STRESS_ONLY_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from case7_gnn_stress_only.runtime import read_config, resolve_device, set_seed
from case7_mlp_stress.trainer import Case7MLPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional MLP stress models on case7 datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path, or use 'auto' for training.save_dir/last.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    if args.resume is not None:
        config.setdefault("training", {})["resume_from"] = args.resume
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["training"]["device"])
    trainer = Case7MLPTrainer(config=config, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()


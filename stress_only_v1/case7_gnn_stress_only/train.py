from __future__ import annotations

import argparse

from case7_gnn_stress_only.runtime import read_config, resolve_device, set_seed
from case7_gnn_stress_only.trainer import Case7Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN models on the case7 dataset.")
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
    trainer = Case7Trainer(config=config, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from case7_node_mlp.runtime import read_config, resolve_device, set_seed
from case7_node_mlp.trainer import NodeMLPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-wise MLP for earpiece MISES PSD density.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))
    device = resolve_device(str(config["training"].get("device", "auto")))
    trainer = NodeMLPTrainer(config=config, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()


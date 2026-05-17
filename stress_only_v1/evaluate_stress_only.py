from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from case7_gnn_stress_only.runtime import ensure_dir, resolve_device, set_seed, write_json
from case7_gnn_stress_only.scalers import StandardScaler
from case7_gnn_stress_only.trainer import Case7Trainer, write_field_diagnostics_csv


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved stress-only checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override evaluation device. Defaults to config training.device.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for eval metrics/diagnostics. Defaults to the checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = checkpoint["config"]

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent)
    config = dict(config)
    config["training"] = dict(config["training"])
    config["training"]["save_dir"] = str(output_dir)

    set_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device or config["training"]["device"])

    trainer = Case7Trainer(config=config, device=device)
    trainer.scalers = {
        name: StandardScaler.from_state_dict(state)
        for name, state in checkpoint["scalers"].items()
    }
    assert trainer.model is not None
    trainer.model.load_state_dict(checkpoint["model_state"])

    split_paths = {
        "train": trainer.train_case_paths,
        "val": trainer.val_case_paths,
        "test": trainer.test_case_paths,
    }
    splits = list(split_paths) if args.split == "all" else [args.split]
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("metrics", {}).get("epoch"),
        "splits": {},
    }

    for split in splits:
        metrics, diagnostics = trainer._evaluate(
            split_paths[split],
            loss_name=trainer.training_cfg["loss"],
            collect_diagnostics=True,
            diagnostic_split=split,
            diagnostic_epoch=checkpoint.get("metrics", {}).get("epoch"),
        )
        summary["splits"][split] = metrics
        write_json(output_dir / f"eval_{split}_metrics.json", metrics)
        write_field_diagnostics_csv(output_dir / f"eval_{split}_diagnostics.csv", diagnostics)

    write_json(output_dir / "eval_metrics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

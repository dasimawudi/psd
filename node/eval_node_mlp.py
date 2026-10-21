"""Standalone evaluation script for a trained NodeMLP checkpoint.

Usage:
  python node/eval_node_mlp.py --checkpoint <path/to/best.pt> [--config <path/to/resolved_config.yaml>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from case7_node_mlp.data import resolve_case_splits, expand_case_sample_paths, discover_case_index
from case7_node_mlp.runtime import read_config, resolve_device, ensure_dir, write_json, make_logger
from case7_node_mlp.trainer import (
    make_loader,
    build_model,
    evaluate,
    write_evaluation_diagnostics,
)
from case7_node_mlp.scalers import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NodeMLP checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to resolved_config.yaml (default: auto-detect from checkpoint directory)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device override (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None, help="Point batch size override")
    parser.add_argument("--sample-batch-size", type=int, default=16, help="Sample batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--splits", type=str, default="val,test", help="Comma-separated splits to evaluate")
    parser.add_argument("--topk-mode", type=str, default="full", help="TopK mode: full, selection, none")
    parser.add_argument("--include-region-metrics", action="store_true", default=True)
    parser.add_argument("--no-region-metrics", dest="include_region_metrics", action="store_false")
    parser.add_argument("--write-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-diagnostics", dest="write_diagnostics", action="store_false")
    parser.add_argument("--max-sample-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for diagnostics output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint_dir = checkpoint_path.parent

    config_path = Path(args.config) if args.config else (checkpoint_dir / "resolved_config.yaml")
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Loading config: {config_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = read_config(str(config_path))
    device = resolve_device(str(args.device)) if args.device != "auto" else resolve_device(
        str(config.get("training", {}).get("device", "auto"))
    )

    # ── Config sections ────────────────────────────────────────────────
    dataset_cfg = dict(config.get("dataset", {}))
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    training_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))

    # ── Scalers & schema from checkpoint ───────────────────────────────
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    feature_schema = dict(checkpoint.get("feature_schema", {}))
    input_dim = int(feature_schema.get("input_dim", 0))

    # ── Build model ────────────────────────────────────────────────────
    model = build_model(config, input_dim=input_dim, feature_schema=feature_schema)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    checkpoint_metrics = checkpoint.get("metrics", {})
    print(f"Checkpoint metrics epoch: {checkpoint_metrics.get('epoch', '?')}")

    # ── Resolve splits ─────────────────────────────────────────────────
    root = str(dataset_cfg.get("root", ".cache/mises_psd_next1000_full_export_step0p5_balanced_v1"))
    case_index = discover_case_index(root)
    split_map = resolve_case_splits(root, dataset_cfg=dataset_cfg)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir

    point_batch_size = args.batch_size or int(training_cfg.get("batch_size", 32768))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split_name in splits:
        if split_name not in split_map:
            print(f"WARNING: split '{split_name}' not found in dataset config, skipping.")
            continue
        print(f"\n{'=' * 60}")
        print(f"Evaluating {split_name} split...")
        print(f"{'=' * 60}")

        case_dirs = [case_index[name] for name in split_map[split_name]]
        sample_paths = expand_case_sample_paths(
            case_dirs, dataset_cfg=dataset_cfg
        )
        if not sample_paths:
            print(f"WARNING: No samples found for {split_name} split, skipping.")
            continue
        print(f"  Samples: {len(sample_paths)}")

        loader = make_loader(
            sample_paths=sample_paths,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
            sample_batch_size=args.sample_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2,
            pin_memory=True,
        )

        logger = make_logger(output_dir, logger_name=f"eval_{split_name}")
        result = evaluate(
            model=model,
            loader=loader,
            y_scaler=y_scaler,
            device=device,
            point_batch_size=point_batch_size,
            split_name=split_name,
            epoch=checkpoint_metrics.get("epoch"),
            logger=logger,
            progress_every_steps=args.progress_every,
            collect_diagnostics=args.write_diagnostics,
            topk_mode=args.topk_mode,
            include_region_metrics=args.include_region_metrics,
            max_sample_batches=args.max_sample_batches,
        )

        print(f"\n  --- {split_name} metrics ---")
        for key, value in sorted(result.metrics.items()):
            print(f"    {key}: {value}")

        if args.write_diagnostics and result.diagnostics:
            diag_path = output_dir / f"eval_{split_name}_diagnostics.csv"
            ensure_dir(output_dir)
            write_evaluation_diagnostics(diag_path, result.diagnostics)
            print(f"\n  Diagnostics saved to: {diag_path}")

        metrics_path = output_dir / f"eval_{split_name}_metrics.json"
        write_json(metrics_path, result.metrics)
        print(f"  Metrics saved to: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

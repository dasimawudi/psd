from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count train-set target-based hotspot/background points.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--output-dir", default="node/outputs/training_region_counts", type=str)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--sample-batch-size", default=None, type=int)
    parser.add_argument("--top-fractions", default="0.01,0.05,0.10,0.25,0.50", type=str)
    parser.add_argument("--global-threshold-sample-size", default=2_000_000, type=int)
    return parser.parse_args()


def _parse_fractions(value: str) -> list[float]:
    fractions = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        fraction = float(part)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"top fraction must be in (0, 1], got {fraction}")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one top fraction is required.")
    return sorted(set(fractions))


def _fraction_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"top{int(round(percent))}pct"
    return f"top{percent:g}pct"


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def main() -> None:
    args = parse_args()
    fractions = _parse_fractions(args.top_fractions)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    training_cfg = dict(config.get("training", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    feature_schema = dict(checkpoint["feature_schema"])

    output_dir = ensure_dir(args.output_dir)
    run_name = args.name or checkpoint_path.parent.name
    logger = make_logger(output_dir, logger_name=f"case7_node_mlp.{run_name}.target_hotspot_count", log_file=f"{run_name}_target_hotspot_count.log")

    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names["train"]]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    loader = make_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=int(args.sample_batch_size or training_cfg.get("sample_batch_size", 1)),
        num_workers=int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)),
        shuffle=False,
    )

    total_points = 0
    positive_points = 0
    target_sum = 0.0
    target_min = float("inf")
    target_max = 0.0
    frame_count = 0
    top_counts = {fraction: 0 for fraction in fractions}
    top_sum = {fraction: 0.0 for fraction in fractions}
    top_min = {fraction: float("inf") for fraction in fractions}
    top_max = {fraction: 0.0 for fraction in fractions}
    top_frames = {fraction: 0 for fraction in fractions}
    reservoir: list[float] = []
    max_reservoir = max(0, int(args.global_threshold_sample_size))
    rng = np.random.default_rng(42)
    started_at = time.monotonic()

    logger.info("Counting train target hotspots | cases=%s | samples=%s", len(case_dirs), len(sample_paths))
    for step, batch in enumerate(loader, start=1):
        target = batch.target_raw.reshape(-1).numpy().astype(np.float64, copy=False)
        sample_index = batch.sample_index.reshape(-1).numpy().astype(np.int64, copy=False)
        total_points += int(target.size)
        positive_points += int((target > 1e-12).sum())
        target_sum += float(target.sum())
        if target.size:
            target_min = min(target_min, float(target.min()))
            target_max = max(target_max, float(target.max()))
            if len(reservoir) < max_reservoir:
                room = max_reservoir - len(reservoir)
                if target.size <= room:
                    reservoir.extend(target.tolist())
                else:
                    reservoir.extend(rng.choice(target, size=room, replace=False).tolist())

        for local_idx in range(len(batch.names)):
            mask = sample_index == local_idx
            n_total = int(mask.sum())
            if n_total <= 0:
                continue
            frame_count += 1
            frame_target = target[mask]
            order = np.argsort(frame_target)[::-1]
            for fraction in fractions:
                count = max(1, int(math.ceil(n_total * fraction)))
                values = frame_target[order[:count]]
                top_counts[fraction] += int(values.size)
                top_sum[fraction] += float(values.sum())
                top_min[fraction] = min(top_min[fraction], float(values.min()))
                top_max[fraction] = max(top_max[fraction], float(values.max()))
                top_frames[fraction] += 1

        if step == 1 or step == len(loader) or step % 200 == 0:
            logger.info("progress | batch=%s/%s | points=%s | elapsed=%s", step, len(loader), total_points, _format_duration(time.monotonic() - started_at))

    rows = []
    for fraction in fractions:
        hotspot_points = int(top_counts[fraction])
        background_points = int(total_points - hotspot_points)
        rows.append(
            {
                "hotspot_definition": f"per-frame target {_fraction_label(fraction)}",
                "top_fraction": fraction,
                "hotspot_points": hotspot_points,
                "background_points": background_points,
                "total_points": int(total_points),
                "hotspot_ratio": float(hotspot_points / max(total_points, 1)),
                "background_ratio": float(background_points / max(total_points, 1)),
                "mean_hotspot_points_per_frame": float(hotspot_points / max(frame_count, 1)),
                "hotspot_target_mean": float(top_sum[fraction] / max(hotspot_points, 1)),
                "hotspot_target_min": float(top_min[fraction]),
                "hotspot_target_max": float(top_max[fraction]),
            }
        )

    thresholds = {}
    reservoir_array = np.asarray(reservoir, dtype=np.float64)
    if reservoir_array.size:
        for fraction in fractions:
            thresholds[f"{_fraction_label(fraction)}_threshold_estimate"] = float(np.quantile(reservoir_array, 1.0 - fraction))

    summary = {
        "name": run_name,
        "checkpoint": str(checkpoint_path),
        "train_cases": len(case_dirs),
        "train_frequency_frames": frame_count,
        "total_case_node_frequency_points": int(total_points),
        "positive_target_points": int(positive_points),
        "target_mean": float(target_sum / max(total_points, 1)),
        "target_min": float(target_min),
        "target_max": float(target_max),
        "hotspot_background_by_per_frame_target_top_fraction": rows,
        "global_target_quantile_threshold_estimates_from_sample": thresholds,
    }
    write_json(output_dir / f"{run_name}_train_target_hotspot_background_summary.json", summary)
    pd.DataFrame(rows).to_csv(output_dir / f"{run_name}_train_target_hotspot_background_summary.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, build_model, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate case-node-frequency metrics for per-frame target top percentages.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--sample-batch-size", default=None, type=int)
    parser.add_argument("--point-batch-size", default=None, type=int)
    parser.add_argument("--top-fractions", default="0.01,0.05,0.10,0.15,0.25", type=str)
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


def _empty_accumulator() -> dict[str, Any]:
    return {"points": 0, "relative_points": 0, "within25": 0, "relative_sum": 0.0, "target_sum": 0.0, "pred_sum": 0.0}


def _row(accumulator: dict[str, Any]) -> dict[str, float | int]:
    relative_points = max(int(accumulator["relative_points"]), 1)
    return {
        "points": int(accumulator["points"]),
        "relative_points": int(accumulator["relative_points"]),
        "within25_ratio": float(accumulator["within25"] / relative_points),
        "relative_mae": float(accumulator["relative_sum"] / relative_points),
        "pred_target_ratio": float(accumulator["pred_sum"] / max(abs(accumulator["target_sum"]), 1e-12)),
    }


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

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"eval_{args.split}_top_percent")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.case_node_f_top_percent", log_file="case_node_f_top_percent.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"]), feature_schema=feature_schema).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[args.split]]
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

    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    overall = _empty_accumulator()
    top_acc = {
        _fraction_label(fraction): {**_empty_accumulator(), "samples": 0, "node_counts": []}
        for fraction in fractions
    }
    started_at = time.monotonic()

    logger.info("Evaluating top percentages | split=%s | cases=%s | samples=%s | device=%s", args.split, len(case_dirs), len(sample_paths), device)
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            pred_parts = []
            for start in range(0, batch.num_points, point_batch_size):
                end = min(start + point_batch_size, batch.num_points)
                pred_parts.append(regression_output(model(batch.features[start:end])))
            pred_scaled = torch.cat(pred_parts, dim=0)
            _pred_log, pred_raw = _decode_prediction(pred_scaled, y_scaler)

            pred = pred_raw.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
            target = batch.target_raw.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
            sample_index = batch.sample_index.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            valid = np.abs(target) > 1e-12
            rel = np.zeros_like(target, dtype=np.float64)
            rel[valid] = np.abs(pred[valid] - target[valid]) / np.abs(target[valid])

            overall["points"] += int(target.size)
            overall["relative_points"] += int(valid.sum())
            overall["within25"] += int((rel[valid] <= 0.25).sum())
            overall["relative_sum"] += float(rel[valid].sum())
            overall["target_sum"] += float(target.sum())
            overall["pred_sum"] += float(pred.sum())

            for local_idx in range(len(batch.names)):
                mask = sample_index == local_idx
                n_total = int(mask.sum())
                if n_total <= 0:
                    continue
                frame_target = target[mask]
                frame_pred = pred[mask]
                frame_valid = np.abs(frame_target) > 1e-12
                frame_rel = np.zeros_like(frame_target, dtype=np.float64)
                frame_rel[frame_valid] = np.abs(frame_pred[frame_valid] - frame_target[frame_valid]) / np.abs(frame_target[frame_valid])
                order = np.argsort(frame_target)[::-1]
                for fraction in fractions:
                    key = _fraction_label(fraction)
                    count = max(1, int(math.ceil(n_total * fraction)))
                    selected = order[:count]
                    selected_valid = frame_valid[selected]
                    acc = top_acc[key]
                    acc["points"] += int(selected.size)
                    acc["relative_points"] += int(selected_valid.sum())
                    acc["within25"] += int((frame_rel[selected][selected_valid] <= 0.25).sum())
                    acc["relative_sum"] += float(frame_rel[selected][selected_valid].sum())
                    acc["target_sum"] += float(frame_target[selected].sum())
                    acc["pred_sum"] += float(frame_pred[selected].sum())
                    acc["samples"] += 1
                    acc["node_counts"].append(count)

            if step == 1 or step == len(loader) or step % int(training_cfg.get("eval_progress_every_steps", 30)) == 0:
                logger.info("progress | batch=%s/%s | points=%s | elapsed=%s", step, len(loader), overall["points"], _format_duration(time.monotonic() - started_at))

    top_rows = {}
    csv_rows = [{"group": "overall", **_row(overall), "samples": len(sample_paths), "mean_nodes_per_frame": ""}]
    for key, acc in top_acc.items():
        row = {**_row(acc), "samples": int(acc["samples"]), "mean_nodes_per_frame": float(np.mean(acc["node_counts"]))}
        top_rows[key] = row
        csv_rows.append({"group": key, **row})

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "cases": len(case_dirs),
        "samples": len(sample_paths),
        "dimension": "case-node-frequency",
        "topk_definition": "per case-frequency frame, top fraction of nodes ranked by target MISES_psd_density",
        "overall": _row(overall),
        "top_fraction_by_case_frequency_target": top_rows,
    }
    write_json(output_dir / "case_node_f_top_percent_metrics.json", summary)
    pd.DataFrame(csv_rows).to_csv(output_dir / "case_node_f_top_percent_metrics.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

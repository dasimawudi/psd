from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

from train7.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from train7.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from train7.scalers import StandardScaler
from train7.trainer import (
    _decode_prediction,
    _effective_prediction_scaled,
    _single_sample_rank_fraction,
    build_model,
    make_loader,
)


FIELDNAMES = [
    "split",
    "sample_name",
    "case_name",
    "frequency_hz",
    "pos_node_index",
    "neg_node_index",
    "pos_target",
    "pos_pred",
    "neg_target",
    "neg_pred",
    "pos_target_rank_fraction",
    "neg_target_rank_fraction",
    "neg_pred_rank_fraction",
    "mode_proximity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V7 resonance hard-negative CSV from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt used for hard-negative mining.")
    parser.add_argument("--config", default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", default="node/outputs/diagnostics/v7_resonance_hard_negatives")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--positive-fraction", type=float, default=0.05)
    parser.add_argument("--negative-target-min-rank", type=float, default=0.50)
    parser.add_argument("--negative-pred-fraction", type=float, default=0.10)
    parser.add_argument("--max-pairs-per-sample", type=int, default=64)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _mode_proximity_hz(frequency_hz: float, mode_frequencies: torch.Tensor | None) -> float:
    if mode_frequencies is None or mode_frequencies.numel() == 0:
        return float("nan")
    return float((mode_frequencies - float(frequency_hz)).abs().min().item())


def _load_mode_frequencies(case_dir: Path) -> torch.Tensor | None:
    path = case_dir / "modal_frequencies.csv"
    if not path.exists():
        return None
    import pandas as pd

    df = pd.read_csv(path)
    if "frequency_hz" in df.columns:
        values = df["frequency_hz"].to_numpy(dtype="float32")
    else:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return None
        values = numeric.iloc[:, -1].to_numpy(dtype="float32")
    return torch.as_tensor(values, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    training_cfg = dict(config.get("training", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    if args.max_cases is not None:
        dataset_cfg["max_cases"] = int(args.max_cases)
        if dataset_cfg.get("split_mode") == "explicit":
            key = f"{args.split}_cases"
            dataset_cfg[key] = list(dataset_cfg.get(key, []))[: int(args.max_cases)]
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir)
    logger = make_logger(output_dir, logger_name="train7.build_hard_negatives", log_file="build_hard_negatives.log")
    device = resolve_device(args.device)
    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    case_index = discover_case_index(dataset_cfg["root"])
    splits = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in splits[args.split]]
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

    mode_cache: dict[str, torch.Tensor | None] = {}
    rows_written = 0
    samples_with_pairs = 0
    csv_path = output_dir / f"{args.split}_hard_negatives.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        with torch.no_grad():
            for step, host_batch in enumerate(loader, start=1):
                predictions: list[torch.Tensor] = []
                point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
                for start in range(0, host_batch.num_points, point_batch_size):
                    stop = min(start + point_batch_size, host_batch.num_points)
                    features = host_batch.features[start:stop].to(device)
                    sample_index = host_batch.sample_index[start:stop].to(device)
                    output = model(features)
                    prediction = _effective_prediction_scaled(
                        output,
                        y_scaler=y_scaler,
                        sample_index=sample_index,
                        peak_relative_cfg=dict(getattr(model, "peak_relative_cfg", {})),
                        zero_gate_cfg=dict(getattr(model, "zero_gate_cfg", {})),
                    )
                    predictions.append(prediction.cpu())
                prediction_scaled = torch.cat(predictions, dim=0)
                pred_log, pred_raw = _decode_prediction(prediction_scaled, y_scaler)

                for sample_idx, sample_name in enumerate(host_batch.names):
                    mask = host_batch.sample_index == sample_idx
                    if not bool(mask.any()):
                        continue
                    target = host_batch.target_raw[mask]
                    pred = pred_raw[mask]
                    local_pred_log = pred_log[mask]
                    target_rank = _single_sample_rank_fraction(target)
                    pred_rank = _single_sample_rank_fraction(local_pred_log)
                    pos_idx = torch.nonzero(target_rank <= float(args.positive_fraction), as_tuple=False).reshape(-1)
                    neg_mask = (target_rank >= float(args.negative_target_min_rank)) & (
                        pred_rank <= float(args.negative_pred_fraction)
                    )
                    neg_idx = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
                    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                        continue
                    samples_with_pairs += 1
                    pos_idx = pos_idx[torch.argsort(target[pos_idx], descending=True)]
                    neg_idx = neg_idx[torch.argsort(pred[neg_idx], descending=True)]
                    max_pairs = max(1, int(args.max_pairs_per_sample))
                    pos_count = max(1, int(math.sqrt(max_pairs)))
                    pos_idx = pos_idx[:pos_count]
                    neg_idx = neg_idx[: max(1, int(math.ceil(max_pairs / max(pos_idx.numel(), 1))))]
                    case_name = host_batch.case_names[sample_idx]
                    if case_name not in mode_cache:
                        mode_cache[case_name] = _load_mode_frequencies(case_index[case_name])
                    mode_proximity = _mode_proximity_hz(float(host_batch.frequency_hz[sample_idx].item()), mode_cache[case_name])
                    node_indices = host_batch.node_indices[mask]
                    pair_count = 0
                    for pidx in pos_idx.tolist():
                        for nidx in neg_idx.tolist():
                            if pair_count >= max_pairs:
                                break
                            writer.writerow(
                                {
                                    "split": args.split,
                                    "sample_name": sample_name,
                                    "case_name": case_name,
                                    "frequency_hz": float(host_batch.frequency_hz[sample_idx].item()),
                                    "pos_node_index": int(node_indices[pidx].item()),
                                    "neg_node_index": int(node_indices[nidx].item()),
                                    "pos_target": float(target[pidx].item()),
                                    "pos_pred": float(pred[pidx].item()),
                                    "neg_target": float(target[nidx].item()),
                                    "neg_pred": float(pred[nidx].item()),
                                    "pos_target_rank_fraction": float(target_rank[pidx].item()),
                                    "neg_target_rank_fraction": float(target_rank[nidx].item()),
                                    "neg_pred_rank_fraction": float(pred_rank[nidx].item()),
                                    "mode_proximity": mode_proximity,
                                }
                            )
                            rows_written += 1
                            pair_count += 1
                        if pair_count >= max_pairs:
                            break
                if step == 1 or step % 50 == 0 or step == len(loader):
                    logger.info(
                        "Hard-negative progress | batch=%s/%s | rows=%s | samples_with_pairs=%s",
                        step,
                        len(loader),
                        rows_written,
                        samples_with_pairs,
                    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "samples": len(sample_paths),
        "rows": rows_written,
        "samples_with_pairs": samples_with_pairs,
        "output_csv": str(csv_path),
    }
    write_json(output_dir / f"{args.split}_summary.json", summary)
    logger.info("Saved hard negatives: %s", summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.hotspot_within25_diagnostics import GroupStats, _load_model
from case7_node_mlp.models import regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, make_loader


DEFAULT_TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate within25 by true target top-k response bands.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument(
        "--top-fractions",
        type=str,
        default=",".join(str(value) for value in DEFAULT_TOP_FRACTIONS),
        help="Comma-separated cumulative top fractions, e.g. 0.01,0.05,0.10,0.25,0.50.",
    )
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
    return sorted(set(fractions))


def _label_fraction(fraction: float) -> str:
    percent = 100.0 * fraction
    if abs(percent - round(percent)) < 1e-8:
        return f"top{int(round(percent))}"
    return f"top{percent:g}"


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1, dtype=np.int64)
    return ranks.astype(np.float32) / float(n)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = [str(row["group"]) for row in rows]
    within25 = [100.0 * float(row["within25_ratio"]) for row in rows]
    p90 = [100.0 * float(row["relative_error_p90"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    bars = ax1.bar(labels, within25, color="#3b7bbf", label="within25")
    ax1.axhline(90.0, color="#222222", linestyle="--", linewidth=1.1, label="90% target")
    ax1.set_ylim(0.0, 100.0)
    ax1.set_ylabel("within25_ratio (%)")
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, within25):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(labels, p90, color="#d64f3a", marker="o", label="raw relative error p90")
    ax2.set_ylabel("raw relative error p90 (%)")
    ax2.set_ylim(0.0, max(100.0, max(p90) * 1.15 if p90 else 100.0))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _update_stats(
    stat: GroupStats,
    mask: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    stat.update(target_raw[mask], pred_raw[mask], target_log[mask], pred_log[mask])


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

    if args.max_cases is not None:
        dataset_cfg["max_cases"] = int(args.max_cases)
        if dataset_cfg.get("split_mode") == "explicit":
            dataset_cfg[args.split + "_cases"] = list(dataset_cfg.get(args.split + "_cases", []))[: int(args.max_cases)]
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"topk_within25_{args.split}")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.topk_within25", log_file="topk_within25.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = _load_model(checkpoint, config, device)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[args.split]]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Running top-k within25 diagnostics | split=%s | cases=%s | samples=%s | fractions=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        ",".join(f"{fraction:g}" for fraction in fractions),
        device,
    )

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

    cumulative_stats = {_label_fraction(fraction): GroupStats(relative_errors=[]) for fraction in fractions}
    cumulative_stats["overall"] = GroupStats(relative_errors=[])
    band_edges = [0.0, *fractions, 1.0]
    band_stats: dict[str, GroupStats] = {}
    for left, right in zip(band_edges[:-1], band_edges[1:]):
        if left == 0.0:
            label = _label_fraction(right)
        elif right == 1.0:
            label = f"{int(round(left * 100))}-100"
        else:
            label = f"{int(round(left * 100))}-{int(round(right * 100))}"
        band_stats[label] = GroupStats(relative_errors=[])

    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    with torch.no_grad():
        for loader_step, host_batch in enumerate(loader, start=1):
            predictions_scaled = []
            for start in range(0, host_batch.num_points, point_batch_size):
                end = min(start + point_batch_size, host_batch.num_points)
                features = host_batch.features[start:end].to(device, non_blocking=True)
                predictions_scaled.append(regression_output(model(features)).detach().cpu())
            prediction_scaled = torch.cat(predictions_scaled, dim=0)
            pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled.to(device), y_scaler)
            pred_log = pred_log_t.cpu().numpy().astype(np.float32, copy=False)
            pred_raw = pred_raw_t.cpu().numpy().astype(np.float32, copy=False)
            target_log = host_batch.target_log.squeeze(-1).numpy().astype(np.float32, copy=False)
            target_raw = host_batch.target_raw.numpy().astype(np.float32, copy=False)
            sample_index = host_batch.sample_index.numpy().astype(np.int64, copy=False)

            rank_fraction = np.ones(target_raw.shape[0], dtype=np.float32)
            for sample_idx in range(len(host_batch.case_names)):
                sample_mask = sample_index == sample_idx
                if sample_mask.any():
                    rank_fraction[sample_mask] = _rank_fraction(target_raw[sample_mask])

            for fraction in fractions:
                label = _label_fraction(fraction)
                _update_stats(cumulative_stats[label], rank_fraction <= fraction, target_raw, pred_raw, target_log, pred_log)
            cumulative_stats["overall"].update(target_raw, pred_raw, target_log, pred_log)

            for (left, right), label in zip(zip(band_edges[:-1], band_edges[1:]), band_stats):
                if right == 1.0:
                    mask = (rank_fraction > left) & (rank_fraction <= right)
                else:
                    mask = (rank_fraction > left) & (rank_fraction <= right)
                _update_stats(band_stats[label], mask, target_raw, pred_raw, target_log, pred_log)

            if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
                logger.info("progress | sample_batch=%s/%s | points=%s", loader_step, len(loader), cumulative_stats["overall"].count)

    cumulative_rows = [cumulative_stats[_label_fraction(fraction)].row(_label_fraction(fraction)) for fraction in fractions]
    cumulative_rows.append(cumulative_stats["overall"].row("overall"))
    band_rows = [stat.row(label) for label, stat in band_stats.items()]

    cumulative_csv = output_dir / "topk_cumulative_within25_summary.csv"
    band_csv = output_dir / "topk_band_within25_summary.csv"
    _write_rows(cumulative_csv, cumulative_rows)
    _write_rows(band_csv, band_rows)
    cumulative_plot = output_dir / "topk_cumulative_within25.png"
    band_plot = output_dir / "topk_band_within25.png"
    _plot_metric(cumulative_plot, cumulative_rows, "Cumulative Top-K Within25 and Raw Relative Error")
    _plot_metric(band_plot, band_rows, "Top-K Bands Within25 and Raw Relative Error")

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "definition": "Per case-frequency nodes ranked by true target_raw; topK means cumulative true top K percent.",
        "top_fractions": fractions,
        "outputs": {
            "cumulative_csv": str(cumulative_csv),
            "band_csv": str(band_csv),
            "cumulative_plot": str(cumulative_plot),
            "band_plot": str(band_plot),
        },
        "cumulative_rows": cumulative_rows,
        "band_rows": band_rows,
    }
    write_json(output_dir / "topk_within25_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.target_quantile_within25_diagnostics import (
    _estimate_quantile_thresholds,
    _format_duration,
    _load_model,
    _selected_case_dirs,
)
from case7_node_mlp.data import expand_case_sample_paths
from case7_node_mlp.trainer import _decode_prediction, make_loader
from case7_node_mlp.models import regression_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot disk-center within25 by 1-percent target quantile band.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="node/outputs/node_mlp_v6_disk_center_baseline/best.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--config", type=str, default=None, help="Config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="node/outputs/node_mlp_v6_disk_center_baseline/target_quantile_within25_p1_all",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument(
        "--max-quantile-values",
        type=int,
        default=5_000_000,
        help="Maximum sampled target values used to estimate p1 thresholds.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from target_quantile_p1_band_within25.csv in output-dir.",
    )
    parser.add_argument("--region-label", type=str, default="Disk-center", help="Region label used in plot titles.")
    return parser.parse_args()


def _format_stress(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-2 or abs_value >= 1e4:
        return f"{value:.2e}"
    if abs_value < 10:
        return f"{value:.2f}"
    if abs_value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _format_stress_range(row: dict[str, Any], index: int) -> str:
    lower = _format_stress(float(row["target_min"]))
    if index >= 99:
        return f">={lower}"
    upper = _format_stress(float(row["target_max"]))
    return f"{lower}-{upper}"


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    mask = denominator > 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _build_rows(thresholds: np.ndarray, accum: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    points = accum["points"]
    relative_points = accum["relative_points"]
    within25_ratio = _safe_divide(accum["within25_count"], relative_points)
    miss25_rate = 1.0 - within25_ratio
    mae = _safe_divide(accum["abs_sum"], points)
    log_mae = _safe_divide(accum["log_abs_sum"], points)
    relative_mae = _safe_divide(accum["relative_sum"], relative_points)
    under_pred_ratio = _safe_divide(accum["under_count"], points)
    over_pred_ratio = _safe_divide(accum["over_count"], points)
    target_mean = _safe_divide(accum["target_sum"], points)
    pred_mean = _safe_divide(accum["pred_sum"], points)
    pred_target_ratio = _safe_divide(accum["pred_sum"], np.maximum(accum["target_sum"], 1e-12))

    rows: list[dict[str, Any]] = []
    for idx in range(100):
        rows.append(
            {
                "group_type": "target_quantile_band_p1",
                "label": f"p{idx}-p{idx + 1}",
                "quantile_start": idx / 100.0,
                "quantile_end": (idx + 1) / 100.0,
                "target_min": float(thresholds[idx]),
                "target_max": float(thresholds[idx + 1]),
                "points": int(points[idx]),
                "relative_points": int(relative_points[idx]),
                "within25_ratio": float(within25_ratio[idx]),
                "miss25_rate": float(miss25_rate[idx]),
                "mae": float(mae[idx]),
                "log_mae": float(log_mae[idx]),
                "relative_mae": float(relative_mae[idx]),
                "under_pred_ratio": float(under_pred_ratio[idx]),
                "over_pred_ratio": float(over_pred_ratio[idx]),
                "target_mean": float(target_mean[idx]),
                "pred_mean": float(pred_mean[idx]),
                "pred_target_ratio": float(pred_target_ratio[idx]),
            }
        )
    return rows


def _plot_p1_within25_labeled(path: Path, rows: list[dict[str, Any]], region_label: str) -> None:
    x = np.arange(len(rows), dtype=np.float64)
    within25 = np.asarray([100.0 * float(row["within25_ratio"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(26, 9.6))
    ax.plot(x, within25, color="#C23B32", linewidth=2.0, marker="o", markersize=3.8)
    ax.axhline(25.0, color="#6B6B6B", linestyle=":", linewidth=1.0)
    ax.set_xlim(-0.8, 99.8)
    ax.set_ylim(0.0, max(100.0, float(np.nanmax(within25)) * 1.12))
    ax.set_xlabel("target quantile band and target value range")
    ax.set_ylabel("within25 (%)")
    ax.set_title(f"{region_label} within25 by p1 target quantile band, all values labeled")
    major_ticks = list(range(0, 100, 5))
    if 99 not in major_ticks:
        major_ticks.append(99)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(
        [f"{rows[idx]['label']}\n{_format_stress_range(rows[idx], idx)}" for idx in major_ticks],
        rotation=38,
        ha="right",
        fontsize=7.2,
    )
    ax.set_xticks(np.arange(0, 100, 1), minor=True)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", which="minor", alpha=0.08)

    for idx, value in enumerate(within25):
        offset = 9 if idx % 2 == 0 else -13
        va = "bottom" if offset > 0 else "top"
        ax.annotate(
            f"{value:.1f}",
            xy=(idx, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=6.8,
            color="#C23B32",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_p1_within25(path: Path, rows: list[dict[str, Any]], region_label: str) -> None:
    x = np.arange(len(rows), dtype=np.float64)
    within25 = np.asarray([100.0 * float(row["within25_ratio"]) for row in rows], dtype=np.float64)
    pred_target = np.asarray([100.0 * float(row["pred_target_ratio"]) for row in rows], dtype=np.float64)
    points = np.asarray([float(row["points"]) for row in rows], dtype=np.float64)
    target_min = np.asarray([float(row["target_min"]) for row in rows], dtype=np.float64)
    target_max = np.asarray([float(row["target_max"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(18, 7.2))
    ax.plot(x, within25, color="#C23B32", linewidth=2.0, marker="o", markersize=3.2, label="within25")
    ax.axhline(25.0, color="#6B6B6B", linestyle=":", linewidth=1.0, label="25% target")
    ax.set_xlim(-0.6, 99.6)
    ax.set_ylim(0.0, max(100.0, float(np.nanmax(within25)) * 1.08))
    ax.set_xlabel("target quantile band and target value range")
    ax.set_ylabel("within25 (%)")
    ax.set_title(f"{region_label} within25 by each 1% target quantile band")
    ax.grid(True, axis="both", alpha=0.23)

    major_ticks = np.arange(0, 100, 5)
    major_labels = [f"{rows[idx]['label']}\n{_format_stress_range(rows[idx], int(idx))}" for idx in major_ticks]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, rotation=38, ha="right", fontsize=7.2)
    ax.set_xticks(np.arange(0, 100, 1), minor=True)
    ax.grid(True, axis="x", which="minor", alpha=0.07)

    for idx in range(0, 100, 5):
        value = within25[idx]
        ax.annotate(
            f"{value:.1f}%",
            xy=(idx, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#C23B32",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )

    ax2 = ax.twinx()
    ax2.plot(x, pred_target, color="#7A5195", linewidth=1.4, alpha=0.65, label="pred/target x100")
    ax2.set_ylabel("pred-target ratio (%)")
    ax2.set_ylim(0.0, max(160.0, float(np.nanmax(pred_target)) * 1.08))

    ax_count = ax.twinx()
    ax_count.spines["right"].set_position(("axes", 1.065))
    ax_count.fill_between(x, 1.0, np.maximum(points, 1.0), color="#5E7F6E", alpha=0.13, label="point count")
    ax_count.set_yscale("log")
    ax_count.set_ylabel("points per p1 band, log")

    handles: list[Any] = []
    labels: list[str] = []
    for axis in (ax, ax2, ax_count):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    ax.legend(handles, labels, loc="upper right")

    stress_notes = [
        f"p0-p1: {_format_stress(target_min[0])}-{_format_stress(target_max[0])}",
        f"p50-p51: {_format_stress(target_min[50])}-{_format_stress(target_max[50])}",
        f"p90-p91: {_format_stress(target_min[90])}-{_format_stress(target_max[90])}",
        f"p99-p100: >={_format_stress(target_min[99])}",
    ]
    ax.text(
        0.01,
        0.02,
        "\n".join(stress_notes),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_p1_tail(path: Path, rows: list[dict[str, Any]], region_label: str) -> None:
    tail_rows = rows[80:]
    x = np.arange(80, 100, dtype=np.float64)
    within25 = np.asarray([100.0 * float(row["within25_ratio"]) for row in tail_rows], dtype=np.float64)
    pred_target = np.asarray([100.0 * float(row["pred_target_ratio"]) for row in tail_rows], dtype=np.float64)
    points = np.asarray([float(row["points"]) for row in tail_rows], dtype=np.float64)
    labels = [str(row["label"]) for row in tail_rows]
    ranges = []
    for row_idx, row in enumerate(tail_rows, start=80):
        lower = _format_stress(float(row["target_min"]))
        upper = _format_stress(float(row["target_max"]))
        ranges.append(f">={lower}" if row_idx == 99 else f"{lower}-{upper}")

    fig, ax = plt.subplots(figsize=(16, 7.5))
    bars = ax.bar(labels, points, color="#5E7F6E", alpha=0.72, label="point count")
    ax.set_yscale("log")
    ax.set_xlabel("target quantile band and stress range")
    ax.set_ylabel("points per p1 band, log")
    ax.set_title(f"{region_label} p80-p100 within25 by each 1% target quantile band")
    ax.set_xticks(np.arange(len(tail_rows)))
    ax.set_xticklabels([f"{label}\n{stress_range}" for label, stress_range in zip(labels, ranges)], rotation=42, ha="right")
    ax.grid(True, axis="y", alpha=0.23)
    for bar, value in zip(bars, points):
        ax.annotate(
            f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax_metric = ax.twinx()
    positions = np.arange(len(tail_rows), dtype=np.float64)
    ax_metric.plot(positions, within25, color="#C23B32", linewidth=2.0, marker="o", label="within25")
    ax_metric.plot(positions, pred_target, color="#7A5195", linewidth=1.5, marker="s", label="pred/target x100")
    ax_metric.axhline(25.0, color="#6B6B6B", linestyle=":", linewidth=1.0)
    ax_metric.set_ylim(0.0, max(160.0, float(np.nanmax(pred_target)) * 1.08))
    ax_metric.set_ylabel("within25 (%) / pred-target ratio (%)")
    for idx, value in enumerate(within25):
        ax_metric.annotate(
            f"{value:.1f}%",
            xy=(idx, value),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#C23B32",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.76},
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_metric.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    csv_path = output_dir / "target_quantile_p1_band_within25.csv"
    plot_path = output_dir / "target_quantile_p1_within25.png"
    tail_plot_path = output_dir / "target_quantile_p1_tail_p80_p100_within25.png"
    labeled_plot_path = output_dir / "target_quantile_p1_within25_labeled.png"
    if args.plot_only:
        rows = _read_rows(csv_path)
        _plot_p1_within25(plot_path, rows, args.region_label)
        _plot_p1_tail(tail_plot_path, rows, args.region_label)
        _plot_p1_within25_labeled(labeled_plot_path, rows, args.region_label)
        summary_path = output_dir / "target_quantile_p1_within25_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}
        outputs = dict(summary.get("outputs", {}))
        outputs.update(
            {
                "p1_band_csv": str(csv_path),
                "p1_within25_plot": str(plot_path),
                "p1_tail_p80_p100_plot": str(tail_plot_path),
                "p1_within25_labeled_plot": str(labeled_plot_path),
            }
        )
        summary["outputs"] = outputs
        write_json(summary_path, summary)
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        return

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    training_cfg = dict(config.get("training", {}))

    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    logger = make_logger(output_dir, logger_name="case7_node_mlp.p1_within25", log_file="p1_within25.log")
    device = resolve_device(args.device)

    case_dirs = _selected_case_dirs(dataset_cfg, args.split)
    if args.max_cases is not None:
        case_dirs = case_dirs[: int(args.max_cases)]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Running p1 within25 diagnostics | split=%s | cases=%s | samples=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        device,
    )

    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    quantiles = [idx / 100.0 for idx in range(101)]
    thresholds, quantile_stats = _estimate_quantile_thresholds(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        quantiles=quantiles,
        num_workers=num_workers,
        max_values=int(args.max_quantile_values),
        logger=logger,
    )

    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    y_scaler_device = y_scaler.to(device)
    model = _load_model(checkpoint, config, device)
    feature_schema = dict(checkpoint["feature_schema"])

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
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=bool(training_cfg.get("persistent_workers", False)),
        prefetch_factor=int(training_cfg.get("prefetch_factor", 2)),
        pin_memory=bool(training_cfg.get("pin_memory", torch.cuda.is_available())),
    )

    accum = {
        "points": np.zeros(100, dtype=np.float64),
        "relative_points": np.zeros(100, dtype=np.float64),
        "within25_count": np.zeros(100, dtype=np.float64),
        "abs_sum": np.zeros(100, dtype=np.float64),
        "log_abs_sum": np.zeros(100, dtype=np.float64),
        "relative_sum": np.zeros(100, dtype=np.float64),
        "under_count": np.zeros(100, dtype=np.float64),
        "over_count": np.zeros(100, dtype=np.float64),
        "target_sum": np.zeros(100, dtype=np.float64),
        "pred_sum": np.zeros(100, dtype=np.float64),
    }

    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    thresholds_for_search = thresholds[1:-1]
    started_at = time.monotonic()
    total_points = 0
    logger.info("Predicting and accumulating p1 buckets | loader_steps=%s | point_batch_size=%s", len(loader), point_batch_size)
    with torch.no_grad():
        for loader_step, host_batch in enumerate(loader, start=1):
            if host_batch.num_points <= 0:
                continue
            prediction_chunks = []
            for start in range(0, host_batch.num_points, point_batch_size):
                stop = min(start + point_batch_size, host_batch.num_points)
                features = host_batch.features[start:stop].to(device, non_blocking=True)
                prediction_chunks.append(regression_output(model(features)).detach())
            prediction_scaled = torch.cat(prediction_chunks, dim=0)
            pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled, y_scaler_device)

            target_raw = host_batch.target_raw.numpy().astype(np.float64, copy=False)
            target_log = host_batch.target_log.squeeze(-1).numpy().astype(np.float64, copy=False)
            pred_raw = pred_raw_t.detach().cpu().numpy().astype(np.float64, copy=False)
            pred_log = pred_log_t.detach().cpu().numpy().astype(np.float64, copy=False)

            finite_mask = np.isfinite(target_raw) & np.isfinite(pred_raw) & np.isfinite(target_log) & np.isfinite(pred_log)
            if not bool(finite_mask.any()):
                continue
            target_raw = target_raw[finite_mask]
            target_log = target_log[finite_mask]
            pred_raw = pred_raw[finite_mask]
            pred_log = pred_log[finite_mask]
            bucket_idx = np.searchsorted(thresholds_for_search, target_raw, side="right")
            bucket_idx = np.clip(bucket_idx, 0, 99)

            delta = pred_raw - target_raw
            abs_error = np.abs(delta)
            log_abs_error = np.abs(pred_log - target_log)
            relative_mask = np.abs(target_raw) > 1e-12
            relative_error = np.zeros_like(abs_error, dtype=np.float64)
            relative_error[relative_mask] = abs_error[relative_mask] / np.abs(target_raw[relative_mask])

            accum["points"] += np.bincount(bucket_idx, minlength=100)
            accum["relative_points"] += np.bincount(bucket_idx[relative_mask], minlength=100)
            accum["within25_count"] += np.bincount(bucket_idx[relative_mask], weights=(relative_error[relative_mask] <= 0.25).astype(np.float64), minlength=100)
            accum["abs_sum"] += np.bincount(bucket_idx, weights=abs_error, minlength=100)
            accum["log_abs_sum"] += np.bincount(bucket_idx, weights=log_abs_error, minlength=100)
            accum["relative_sum"] += np.bincount(bucket_idx[relative_mask], weights=relative_error[relative_mask], minlength=100)
            accum["under_count"] += np.bincount(bucket_idx, weights=(pred_raw < target_raw).astype(np.float64), minlength=100)
            accum["over_count"] += np.bincount(bucket_idx, weights=(pred_raw > target_raw).astype(np.float64), minlength=100)
            accum["target_sum"] += np.bincount(bucket_idx, weights=target_raw, minlength=100)
            accum["pred_sum"] += np.bincount(bucket_idx, weights=pred_raw, minlength=100)
            total_points += int(target_raw.size)

            if loader_step == 1 or loader_step == len(loader) or loader_step % 30 == 0:
                logger.info(
                    "predict progress | sample_batch=%s/%s | points=%s | elapsed=%s",
                    loader_step,
                    len(loader),
                    total_points,
                    _format_duration(time.monotonic() - started_at),
                )

    rows = _build_rows(thresholds, accum)
    _write_rows(csv_path, rows)
    _plot_p1_within25(plot_path, rows, args.region_label)
    _plot_p1_tail(tail_plot_path, rows, args.region_label)
    _plot_p1_within25_labeled(labeled_plot_path, rows, args.region_label)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "cases": len(case_dirs),
        "samples": len(sample_paths),
        "points": int(total_points),
        "quantiles": quantiles,
        "thresholds": [float(value) for value in thresholds],
        "quantile_threshold_estimation": quantile_stats,
        "outputs": {
            "p1_band_csv": str(csv_path),
            "p1_within25_plot": str(plot_path),
            "p1_tail_p80_p100_plot": str(tail_plot_path),
            "p1_within25_labeled_plot": str(labeled_plot_path),
        },
        "band_rows": rows,
    }
    write_json(output_dir / "target_quantile_p1_within25_summary.json", summary)
    logger.info("Saved p1 within25 diagnostics: %s", output_dir)
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

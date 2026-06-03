from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import PointMLP
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, _load_target_values_for_stats, make_loader


DEFAULT_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999)


@dataclass
class BucketStats:
    points: int = 0
    relative_points: int = 0
    within25_count: int = 0
    abs_sum: float = 0.0
    log_abs_sum: float = 0.0
    relative_sum: float = 0.0
    under_count: int = 0
    over_count: int = 0
    target_sum: float = 0.0
    pred_sum: float = 0.0

    def update(self, target_raw: np.ndarray, pred_raw: np.ndarray, target_log: np.ndarray, pred_log: np.ndarray) -> None:
        if target_raw.size == 0:
            return
        delta = pred_raw - target_raw
        abs_error = np.abs(delta)
        log_abs_error = np.abs(pred_log - target_log)
        relative_mask = np.abs(target_raw) > 1e-12
        relative_error = np.zeros_like(abs_error, dtype=np.float64)
        relative_error[relative_mask] = abs_error[relative_mask] / np.abs(target_raw[relative_mask])

        self.points += int(target_raw.size)
        self.relative_points += int(relative_mask.sum())
        self.within25_count += int((relative_error[relative_mask] <= 0.25).sum())
        self.abs_sum += float(abs_error.sum())
        self.log_abs_sum += float(log_abs_error.sum())
        self.relative_sum += float(relative_error[relative_mask].sum())
        self.under_count += int((pred_raw < target_raw).sum())
        self.over_count += int((pred_raw > target_raw).sum())
        self.target_sum += float(target_raw.sum())
        self.pred_sum += float(pred_raw.sum())

    def row(self, group_type: str, label: str, quantile: float | None, threshold: float | None) -> dict[str, Any]:
        points = max(self.points, 1)
        relative_points = max(self.relative_points, 1)
        within25 = self.within25_count / relative_points
        return {
            "group_type": group_type,
            "label": label,
            "quantile": "" if quantile is None else quantile,
            "target_threshold": "" if threshold is None else threshold,
            "points": self.points,
            "relative_points": self.relative_points,
            "within25_ratio": within25,
            "miss25_rate": 1.0 - within25,
            "mae": self.abs_sum / points,
            "log_mae": self.log_abs_sum / points,
            "relative_mae": self.relative_sum / relative_points,
            "under_pred_ratio": self.under_count / points,
            "over_pred_ratio": self.over_count / points,
            "target_mean": self.target_sum / points,
            "pred_mean": self.pred_sum / points,
            "pred_target_ratio": self.pred_sum / max(self.target_sum, 1e-12),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count target quantile tails and within25 for a node MLP checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument(
        "--quantiles",
        type=str,
        default=",".join(str(value) for value in DEFAULT_QUANTILES),
        help="Comma-separated target quantiles used as cumulative thresholds.",
    )
    parser.add_argument(
        "--max-quantile-values",
        type=int,
        default=5_000_000,
        help="Maximum target values sampled to estimate quantile thresholds.",
    )
    return parser.parse_args()


def _parse_quantiles(value: str) -> list[float]:
    quantiles = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        quantile = float(part)
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        quantiles.append(quantile)
    if not quantiles:
        raise ValueError("At least one quantile is required.")
    return sorted(set(quantiles))


def _quantile_label(quantile: float) -> str:
    percent = quantile * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"p{int(round(percent))}"
    return f"p{percent:g}"


def _format_duration(seconds: float) -> str:
    seconds_int = int(round(seconds))
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _selected_case_dirs(dataset_cfg: dict[str, Any], split: str) -> list[Path]:
    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    if split == "all":
        case_names: list[str] = []
        seen: set[str] = set()
        for split_name in ("train", "val", "test"):
            for case_name in split_names.get(split_name, []):
                if case_name not in seen:
                    seen.add(case_name)
                    case_names.append(case_name)
    else:
        case_names = list(split_names[split])
    return [case_index[name] for name in case_names]


def _sample_for_quantiles(path: Path, dataset_cfg: dict[str, Any], per_file_cap: int | None) -> np.ndarray:
    values = _load_target_values_for_stats(path, dataset_cfg)
    if per_file_cap is not None and values.size > per_file_cap:
        seed = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
        indices = np.random.default_rng(seed).choice(values.size, size=per_file_cap, replace=False)
        values = values[indices]
    return values.astype(np.float64, copy=False)


def _estimate_quantile_thresholds(
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    quantiles: list[float],
    num_workers: int,
    max_values: int,
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    started_at = time.monotonic()
    workers = max(1, int(num_workers))
    per_file_cap = None
    if max_values > 0:
        per_file_cap = max(1, int(max_values) // max(len(sample_paths), 1))
    logger.info(
        "Estimating target quantile thresholds | samples=%s | workers=%s | per_file_cap=%s",
        len(sample_paths),
        workers,
        per_file_cap,
    )

    values: list[np.ndarray] = []
    sampled_points = 0
    scanned_points = 0

    def consume(array: np.ndarray) -> None:
        nonlocal sampled_points, scanned_points
        scanned_points += int(array.size)
        sampled_points += int(array.size)
        values.append(array)

    if workers == 1:
        for idx, path in enumerate(sample_paths, start=1):
            consume(_sample_for_quantiles(path, dataset_cfg, per_file_cap))
            if idx == 1 or idx == len(sample_paths) or idx % 1000 == 0:
                logger.info(
                    "quantile progress | sample=%s/%s | sampled_values=%s | elapsed=%s",
                    idx,
                    len(sample_paths),
                    sampled_points,
                    _format_duration(time.monotonic() - started_at),
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            path_iter = iter(sample_paths)
            pending: dict[Any, Path] = {}
            max_pending = max(workers, workers * 4)

            def submit_until_full() -> None:
                while len(pending) < max_pending:
                    try:
                        path = next(path_iter)
                    except StopIteration:
                        return
                    pending[executor.submit(_sample_for_quantiles, path, dataset_cfg, per_file_cap)] = path

            submit_until_full()
            completed = 0
            while pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    completed += 1
                    consume(future.result())
                if completed == 1 or completed == len(sample_paths) or completed % 1000 == 0:
                    logger.info(
                        "quantile progress | sample=%s/%s | sampled_values=%s | elapsed=%s",
                        completed,
                        len(sample_paths),
                        sampled_points,
                        _format_duration(time.monotonic() - started_at),
                    )
                submit_until_full()

    if not values:
        raise ValueError("No target values were available for quantile threshold estimation.")
    concatenated = np.concatenate(values)
    thresholds = np.quantile(concatenated, np.asarray(quantiles, dtype=np.float64))
    stats = {
        "sampled_values": int(concatenated.size),
        "sampled_min": float(np.min(concatenated)),
        "sampled_max": float(np.max(concatenated)),
        "elapsed_seconds": float(time.monotonic() - started_at),
    }
    return thresholds.astype(np.float64, copy=False), stats


def _load_model(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device) -> PointMLP:
    feature_schema = dict(checkpoint["feature_schema"])
    model_cfg = dict(config.get("model", {}))
    model = PointMLP(
        input_dim=int(feature_schema["input_dim"]),
        hidden_dims=[int(dim) for dim in model_cfg.get("hidden_dims", [256, 256, 128])],
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "silu")),
        use_layer_norm=bool(model_cfg.get("layer_norm", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_cumulative(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = [str(row["label"]) for row in rows]
    counts = [int(row["points"]) for row in rows]
    within25 = [100.0 * float(row["within25_ratio"]) for row in rows]

    fig, ax_count = plt.subplots(figsize=(11, 5.5))
    bars = ax_count.bar(labels, counts, color="#5279a7", alpha=0.80, label="point count")
    ax_count.set_yscale("log")
    ax_count.set_ylabel("points with target >= quantile threshold (log)")
    ax_count.set_xlabel("target quantile threshold")
    ax_count.set_title(title)
    ax_count.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        ax_count.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(count, 1),
            f"{count/1e6:.1f}M" if count >= 1_000_000 else str(count),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax_within = ax_count.twinx()
    ax_within.plot(labels, within25, color="#c9473d", marker="o", linewidth=2.0, label="within25")
    ax_within.set_ylim(0.0, 100.0)
    ax_within.set_ylabel("within25 ratio (%)")
    for label, value in zip(labels, within25):
        ax_within.annotate(f"{value:.1f}%", xy=(label, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)

    handles1, labels1 = ax_count.get_legend_handles_labels()
    handles2, labels2 = ax_within.get_legend_handles_labels()
    ax_count.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_band(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = [str(row["label"]) for row in rows]
    counts = [int(row["relative_points"]) for row in rows]
    within25 = [100.0 * float(row["within25_ratio"]) for row in rows]

    fig, ax_count = plt.subplots(figsize=(12, 5.5))
    ax_count.bar(labels, counts, color="#6f8f72", alpha=0.80, label="point count")
    ax_count.set_yscale("log")
    ax_count.set_ylabel("points in quantile band (log)")
    ax_count.set_xlabel("target quantile band")
    ax_count.set_title(title)
    ax_count.grid(axis="y", alpha=0.25)

    ax_within = ax_count.twinx()
    ax_within.plot(labels, within25, color="#c9473d", marker="o", linewidth=2.0, label="within25")
    ax_within.set_ylim(0.0, 100.0)
    ax_within.set_ylabel("within25 ratio (%)")

    handles1, labels1 = ax_count.get_legend_handles_labels()
    handles2, labels2 = ax_within.get_legend_handles_labels()
    ax_count.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    training_cfg = dict(config.get("training", {}))
    quantiles = _parse_quantiles(args.quantiles)

    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    case_dirs = _selected_case_dirs(dataset_cfg, args.split)
    if args.max_cases is not None:
        case_dirs = case_dirs[: int(args.max_cases)]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"target_quantile_within25_{args.split}")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.target_quantile_within25", log_file="target_quantile_within25.log")
    device = resolve_device(args.device)
    logger.info(
        "Running target quantile within25 diagnostics | split=%s | cases=%s | samples=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        device,
    )

    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    thresholds, quantile_stats = _estimate_quantile_thresholds(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        quantiles=quantiles,
        num_workers=num_workers,
        max_values=int(args.max_quantile_values),
        logger=logger,
    )
    for quantile, threshold in zip(quantiles, thresholds):
        logger.info("target threshold | %s | %.8g", _quantile_label(quantile), float(threshold))

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = _load_model(checkpoint, config, device)

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

    cumulative_stats = {
        _quantile_label(quantile): BucketStats() for quantile in quantiles
    }
    band_stats: dict[str, BucketStats] = {}
    for left, right in zip(quantiles[:-1], quantiles[1:]):
        band_stats[f"{_quantile_label(left)}-{_quantile_label(right)}"] = BucketStats()

    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    started_at = time.monotonic()
    total_points = 0
    logger.info("Predicting and accumulating buckets | loader_steps=%s | point_batch_size=%s", len(loader), point_batch_size)
    with torch.no_grad():
        for loader_step, host_batch in enumerate(loader, start=1):
            if host_batch.num_points <= 0:
                continue
            predictions_scaled = []
            for start in range(0, host_batch.num_points, point_batch_size):
                stop = min(start + point_batch_size, host_batch.num_points)
                features = host_batch.features[start:stop].to(device, non_blocking=True)
                predictions_scaled.append(model(features).detach().cpu())
            prediction_scaled = torch.cat(predictions_scaled, dim=0)
            pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled.to(device), y_scaler)
            pred_log = pred_log_t.cpu().numpy().astype(np.float64, copy=False)
            pred_raw = pred_raw_t.cpu().numpy().astype(np.float64, copy=False)
            target_log = host_batch.target_log.squeeze(-1).numpy().astype(np.float64, copy=False)
            target_raw = host_batch.target_raw.numpy().astype(np.float64, copy=False)
            total_points += int(target_raw.size)

            for quantile, threshold in zip(quantiles, thresholds):
                label = _quantile_label(quantile)
                mask = target_raw >= float(threshold)
                cumulative_stats[label].update(target_raw[mask], pred_raw[mask], target_log[mask], pred_log[mask])

            for idx, label in enumerate(band_stats):
                left = float(thresholds[idx])
                right = float(thresholds[idx + 1])
                if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                    mask = (target_raw >= left) & (target_raw <= right)
                else:
                    mask = (target_raw >= left) & (target_raw < right)
                band_stats[label].update(target_raw[mask], pred_raw[mask], target_log[mask], pred_log[mask])

            if loader_step == 1 or loader_step == len(loader) or loader_step % 30 == 0:
                logger.info(
                    "predict progress | sample_batch=%s/%s | points=%s | elapsed=%s",
                    loader_step,
                    len(loader),
                    total_points,
                    _format_duration(time.monotonic() - started_at),
                )

    cumulative_rows = [
        cumulative_stats[_quantile_label(quantile)].row(
            group_type="target_ge_quantile",
            label=_quantile_label(quantile),
            quantile=quantile,
            threshold=float(threshold),
        )
        for quantile, threshold in zip(quantiles, thresholds)
    ]
    band_rows = []
    for idx, (label, stats) in enumerate(band_stats.items()):
        band_rows.append(
            stats.row(
                group_type="target_quantile_band",
                label=label,
                quantile=None,
                threshold=None,
            )
        )

    cumulative_csv = output_dir / "target_quantile_cumulative_within25.csv"
    band_csv = output_dir / "target_quantile_band_within25.csv"
    cumulative_plot = output_dir / "target_quantile_cumulative_within25.png"
    band_plot = output_dir / "target_quantile_band_within25.png"
    _write_rows(cumulative_csv, cumulative_rows)
    _write_rows(band_csv, band_rows)
    _plot_cumulative(cumulative_plot, cumulative_rows, f"Target Quantile Count and Within25 ({args.split})")
    _plot_band(band_plot, band_rows, f"Target Quantile Band Count and Within25 ({args.split})")

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "cases": len(case_dirs),
        "samples": len(sample_paths),
        "points": total_points,
        "quantiles": quantiles,
        "thresholds": [float(value) for value in thresholds],
        "quantile_threshold_estimation": quantile_stats,
        "outputs": {
            "cumulative_csv": str(cumulative_csv),
            "band_csv": str(band_csv),
            "cumulative_plot": str(cumulative_plot),
            "band_plot": str(band_plot),
        },
        "cumulative_rows": cumulative_rows,
        "band_rows": band_rows,
    }
    write_json(output_dir / "target_quantile_within25_summary.json", summary)
    logger.info("Saved target quantile diagnostics: %s", output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

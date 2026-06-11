from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, build_model, make_loader


FREQUENCY_BINS = [20.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0, float("inf")]
POINT_TARGET_BINS = [0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, float("inf")]
SAMPLE_TARGET_MEAN_BINS = [0.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, float("inf")]
REGION_LABELS = [
    "background_non_stress_region",
    "center_couple_region",
    "plate_hole_region",
    "ear_hole_region",
    "ear_connection_region",
    "bc_region",
]
MODE_PROXIMITY_LABELS = ["near_mode_2pct", "near_mode_5pct", "far_from_mode_5pct", "no_mode_data"]
REGION_COLUMNS = {
    "center_couple_region": ("center_couple_mask",),
    "plate_hole_region": ("plate_hole_wall_mask",),
    "ear_hole_region": ("ear_hole_wall_mask",),
    "ear_connection_region": (
        "ear_connection_fillet_mask",
        "ear_connection_earside_mask",
        "ear_connection_mask",
    ),
    "bc_region": ("bc_mask",),
}


@dataclass
class BucketStats:
    count: int = 0
    relative_count: int = 0
    within25_count: int = 0
    abs_sum: float = 0.0
    log_abs_sum: float = 0.0
    relative_sum: float = 0.0
    symmetric_relative_sum: float = 0.0
    under_count: int = 0
    over_count: int = 0
    target_sum: float = 0.0
    pred_sum: float = 0.0

    def update(
        self,
        target_raw: np.ndarray,
        pred_raw: np.ndarray,
        target_log: np.ndarray,
        pred_log: np.ndarray,
    ) -> None:
        if target_raw.size == 0:
            return
        delta = pred_raw - target_raw
        abs_error = np.abs(delta)
        log_abs = np.abs(pred_log - target_log)
        relative_mask = np.abs(target_raw) > 1e-12
        relative = np.zeros_like(abs_error, dtype=np.float64)
        relative[relative_mask] = abs_error[relative_mask] / np.abs(target_raw[relative_mask])
        symmetric = abs_error / np.maximum(0.5 * (np.abs(pred_raw) + np.abs(target_raw)), 1e-12)

        self.count += int(target_raw.size)
        self.relative_count += int(relative_mask.sum())
        self.within25_count += int((relative[relative_mask] <= 0.25).sum())
        self.abs_sum += float(abs_error.sum())
        self.log_abs_sum += float(log_abs.sum())
        self.relative_sum += float(relative[relative_mask].sum())
        self.symmetric_relative_sum += float(symmetric.sum())
        self.under_count += int((pred_raw < target_raw).sum())
        self.over_count += int((pred_raw > target_raw).sum())
        self.target_sum += float(target_raw.sum())
        self.pred_sum += float(pred_raw.sum())

    def row(self, bucket_type: str, bucket: str) -> dict[str, Any]:
        points = max(self.count, 1)
        relative_points = max(self.relative_count, 1)
        return {
            "bucket_type": bucket_type,
            "bucket": bucket,
            "points": self.count,
            "relative_points": self.relative_count,
            "within25_ratio": self.within25_count / relative_points,
            "miss25_rate": 1.0 - self.within25_count / relative_points,
            "mae": self.abs_sum / points,
            "log_mae": self.log_abs_sum / points,
            "relative_mae": self.relative_sum / relative_points,
            "symmetric_relative_mae": self.symmetric_relative_sum / points,
            "under_pred_ratio": self.under_count / points,
            "over_pred_ratio": self.over_count / points,
            "target_mean": self.target_sum / points,
            "pred_mean": self.pred_sum / points,
            "pred_target_ratio": self.pred_sum / max(self.target_sum, 1e-12),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-level diagnostics for raw-scale within25_ratio.")
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
    return parser.parse_args()


def _format_edge(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if abs(value) >= 10000.0:
        return f"{value:.0e}"
    return f"{value:g}"


def _labels(edges: list[float]) -> list[str]:
    return [f"[{_format_edge(left)}, {_format_edge(right)})" for left, right in zip(edges[:-1], edges[1:])]


def _bin_indices(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") - 1, 0, len(edges) - 2)


def _update_bucket_group(
    buckets: dict[str, BucketStats],
    bucket_type: str,
    labels: list[str],
    ids: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    for bucket_id in np.unique(ids):
        mask = ids == bucket_id
        label = labels[int(bucket_id)]
        key = f"{bucket_type}:{label}"
        buckets.setdefault(key, BucketStats()).update(
            target_raw=target_raw[mask],
            pred_raw=pred_raw[mask],
            target_log=target_log[mask],
            pred_log=pred_log[mask],
        )


def _case_region_arrays(case_dir: Path) -> dict[str, np.ndarray]:
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    arrays: dict[str, np.ndarray] = {}
    stress_any = np.zeros(len(nodes_df), dtype=bool)
    for label, columns in REGION_COLUMNS.items():
        mask = np.zeros(len(nodes_df), dtype=bool)
        for column in columns:
            if column in nodes_df.columns:
                mask |= nodes_df[column].to_numpy(dtype=np.float32) > 0.5
        arrays[label] = mask
        if label != "bc_region":
            stress_any |= mask
    arrays["background_non_stress_region"] = ~stress_any & ~arrays.get("bc_region", np.zeros(len(nodes_df), dtype=bool))
    return arrays


def _case_mode_frequencies(case_dir: Path) -> np.ndarray:
    path = case_dir / "modal_frequencies.csv"
    if not path.exists():
        return np.array([], dtype=np.float32)
    df = pd.read_csv(path)
    if "frequency_hz" in df.columns:
        return df["frequency_hz"].to_numpy(dtype=np.float32)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.array([], dtype=np.float32)
    return numeric.iloc[:, -1].to_numpy(dtype=np.float32)


def _mode_proximity_id(mode_frequencies: np.ndarray, frequency_hz: float) -> int:
    if mode_frequencies.size == 0:
        return MODE_PROXIMITY_LABELS.index("no_mode_data")
    relative_delta = float(np.min(np.abs(mode_frequencies - float(frequency_hz))) / max(float(frequency_hz), 1e-12))
    if relative_delta <= 0.02:
        return MODE_PROXIMITY_LABELS.index("near_mode_2pct")
    if relative_delta <= 0.05:
        return MODE_PROXIMITY_LABELS.index("near_mode_5pct")
    return MODE_PROXIMITY_LABELS.index("far_from_mode_5pct")


def _exclusive_region_ids(case_regions: dict[str, np.ndarray], node_rows: np.ndarray) -> np.ndarray:
    ids = np.zeros(node_rows.shape[0], dtype=np.int16)
    # Priority keeps hotspot-related masks from being hidden by broader masks.
    for label in ("center_couple_region", "plate_hole_region", "ear_hole_region", "ear_connection_region", "bc_region"):
        mask = case_regions.get(label, np.zeros(0, dtype=bool))[node_rows]
        ids[mask] = REGION_LABELS.index(label)
    return ids


def _write_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket_type",
        "bucket",
        "points",
        "relative_points",
        "within25_ratio",
        "miss25_rate",
        "mae",
        "log_mae",
        "relative_mae",
        "symmetric_relative_mae",
        "under_pred_ratio",
        "over_pred_ratio",
        "target_mean",
        "pred_mean",
        "pred_target_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cross_bucket_rows(
    *,
    row_type: str,
    row_labels: list[str],
    row_ids: np.ndarray,
    col_type: str,
    col_labels: list[str],
    col_ids: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            mask = (row_ids == row_idx) & (col_ids == col_idx)
            if not bool(mask.any()):
                continue
            summary = _summary_from_arrays(target_raw[mask], pred_raw[mask], target_log[mask], pred_log[mask])
            rows.append(
                {
                    "row_type": row_type,
                    "row_bucket": row_label,
                    "col_type": col_type,
                    "col_bucket": col_label,
                    "points": summary["points"],
                    "relative_points": summary["relative_points"],
                    "within25_ratio": summary["within25_ratio"],
                    "miss25_rate": 1.0 - float(summary["within25_ratio"]),
                    "log_mae": summary["log_mae"],
                    "relative_mae": summary["relative_mae"],
                    "target_mean": float(np.mean(target_raw[mask])),
                    "pred_mean": float(np.mean(pred_raw[mask])),
                    "pred_target_ratio": float(np.sum(pred_raw[mask]) / max(float(np.sum(target_raw[mask])), 1e-12)),
                }
            )
    return rows


def _write_cross_bucket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "row_bucket",
        "col_type",
        "col_bucket",
        "points",
        "relative_points",
        "within25_ratio",
        "miss25_rate",
        "log_mae",
        "relative_mae",
        "target_mean",
        "pred_mean",
        "pred_target_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_error_distribution(
    path: Path,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    abs_error = np.abs(pred_raw - target_raw)
    relative_mask = np.abs(target_raw) > 1e-12
    relative = abs_error[relative_mask] / np.abs(target_raw[relative_mask])
    log_abs = np.abs(pred_log - target_log)
    sample_size = min(int(relative.size), 1_000_000)
    rng = np.random.default_rng(42)
    if relative.size > sample_size:
        indices = rng.choice(relative.size, size=sample_size, replace=False)
        relative_sample = relative[indices]
    else:
        relative_sample = relative
    if log_abs.size > sample_size:
        indices = rng.choice(log_abs.size, size=sample_size, replace=False)
        log_sample = log_abs[indices]
    else:
        log_sample = log_abs

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(np.clip(relative_sample, 0.0, 2.0), bins=100, color="#4C78A8", alpha=0.85)
    axes[0].axvline(0.25, color="#D62728", linestyle="--", linewidth=1.4, label="25% threshold")
    axes[0].set_xlabel("raw relative error, clipped at 2")
    axes[0].set_ylabel("sampled points")
    axes[0].set_title("Relative Error Distribution")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].hist(np.clip(log_sample, 0.0, 2.0), bins=100, color="#59A14F", alpha=0.85)
    axes[1].set_xlabel("absolute log error, clipped at 2")
    axes[1].set_ylabel("sampled points")
    axes[1].set_title("Log Error Distribution")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error_cdf(path: Path, target_raw: np.ndarray, pred_raw: np.ndarray) -> None:
    abs_error = np.abs(pred_raw - target_raw)
    relative_mask = np.abs(target_raw) > 1e-12
    relative = abs_error[relative_mask] / np.abs(target_raw[relative_mask])
    if relative.size == 0:
        return
    quantiles = np.linspace(0.0, 0.999, 400)
    values = np.quantile(relative, quantiles)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, quantiles, linewidth=2.0, color="#4C78A8")
    ax.axvline(0.25, color="#D62728", linestyle="--", linewidth=1.4, label="25% threshold")
    ax.axhline(0.90, color="#F58518", linestyle="--", linewidth=1.4, label="90% target")
    ax.set_xlim(0.0, min(max(float(np.quantile(relative, 0.99)), 0.5), 3.0))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("raw relative error")
    ax.set_ylabel("fraction of points <= error")
    ax.set_title("Relative Error CDF")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_bucket_table(path: Path, csv_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(max(9, len(df) * 1.2), 5))
    x = np.arange(len(df))
    ax1.bar(x, df["within25_ratio"].astype(float), color="#4C78A8", alpha=0.85)
    ax1.axhline(0.90, color="#D62728", linestyle="--", linewidth=1.4, label="90% target")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("within25_ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["bucket"].astype(str), rotation=35, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, df["points"].astype(float), color="#F58518", marker="o", linewidth=1.5, label="points")
    ax2.set_ylabel("points")
    ax1.set_title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cross_bucket_heatmap(path: Path, csv_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    row_labels = list(dict.fromkeys(df["row_bucket"].astype(str).tolist()))
    col_labels = list(dict.fromkeys(df["col_bucket"].astype(str).tolist()))
    values = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float32)
    points = np.zeros((len(row_labels), len(col_labels)), dtype=np.float64)
    row_lookup = {label: idx for idx, label in enumerate(row_labels)}
    col_lookup = {label: idx for idx, label in enumerate(col_labels)}
    for item in df.itertuples(index=False):
        row_idx = row_lookup[str(item.row_bucket)]
        col_idx = col_lookup[str(item.col_bucket)]
        values[row_idx, col_idx] = float(item.within25_ratio)
        points[row_idx, col_idx] = float(item.points)

    fig_width = max(10, 1.0 * len(col_labels))
    fig_height = max(4, 0.7 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, vmin=0.0, vmax=0.9, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel(str(df["col_type"].iloc[0]))
    ax.set_ylabel(str(df["row_type"].iloc[0]))
    for row_idx in range(len(row_labels)):
        for col_idx in range(len(col_labels)):
            if np.isfinite(values[row_idx, col_idx]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{values[row_idx, col_idx]:.2f}\n{points[row_idx, col_idx] / 1e6:.1f}M",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("within25_ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_calibration(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x, df["within25_ratio"].astype(float), color="#4C78A8", alpha=0.85)
    ax.axhline(0.90, color="#D62728", linestyle="--", linewidth=1.4, label="90% target")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("within25_ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"].astype(str), rotation=25, ha="right")
    ax.set_title("Oracle Log-Bias Calibration")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary_from_arrays(target_raw: np.ndarray, pred_raw: np.ndarray, target_log: np.ndarray, pred_log: np.ndarray) -> dict[str, Any]:
    abs_error = np.abs(pred_raw - target_raw)
    relative_mask = np.abs(target_raw) > 1e-12
    relative = abs_error[relative_mask] / np.abs(target_raw[relative_mask])
    log_abs = np.abs(pred_log - target_log)
    symmetric = abs_error / np.maximum(0.5 * (np.abs(pred_raw) + np.abs(target_raw)), 1e-12)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "points": int(target_raw.size),
        "relative_points": int(relative_mask.sum()),
        "within25_ratio": float((relative <= 0.25).mean()) if relative.size else 0.0,
        "mae": float(abs_error.mean()) if abs_error.size else 0.0,
        "log_mae": float(log_abs.mean()) if log_abs.size else 0.0,
        "relative_mae": float(relative.mean()) if relative.size else 0.0,
        "symmetric_relative_mae": float(symmetric.mean()) if symmetric.size else 0.0,
        "under_pred_ratio": float((pred_raw < target_raw).mean()) if target_raw.size else 0.0,
        "pred_target_ratio": float(pred_raw.sum() / max(float(target_raw.sum()), 1e-12)),
        "relative_error_quantiles": {str(q): float(np.quantile(relative, q)) for q in quantiles} if relative.size else {},
        "log_abs_error_quantiles": {str(q): float(np.quantile(log_abs, q)) for q in quantiles} if log_abs.size else {},
        "symmetric_relative_error_quantiles": {
            str(q): float(np.quantile(symmetric, q)) for q in quantiles
        }
        if symmetric.size
        else {},
    }


def _apply_log_bias_calibration(
    target_raw: np.ndarray,
    pred_log: np.ndarray,
    target_log: np.ndarray,
    group_ids: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    calibrated_log = pred_log.astype(np.float64, copy=True)
    mask = target_raw > 1e-12
    biases: dict[str, float] = {}
    if group_ids is None:
        bias = float(np.median((target_log - pred_log)[mask])) if mask.any() else 0.0
        calibrated_log += bias
        biases["all"] = bias
        return np.expm1(calibrated_log).clip(min=0.0).astype(np.float32), biases

    for group_id in np.unique(group_ids):
        group_mask = (group_ids == group_id) & mask
        bias = float(np.median((target_log - pred_log)[group_mask])) if group_mask.any() else 0.0
        calibrated_log[group_ids == group_id] += bias
        biases[str(int(group_id))] = bias
    return np.expm1(calibrated_log).clip(min=0.0).astype(np.float32), biases


def _calibration_rows(
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    pred_log: np.ndarray,
    target_log: np.ndarray,
    frequency_ids: np.ndarray,
    point_target_ids: np.ndarray,
    sample_mean_ids: np.ndarray,
    region_ids: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = [
        ("none", None),
        ("global_log_bias", None),
        ("frequency_bucket_log_bias", frequency_ids),
        ("point_target_bucket_log_bias_oracle", point_target_ids),
        ("sample_target_mean_bucket_log_bias_oracle", sample_mean_ids),
        ("region_log_bias", region_ids),
    ]
    for name, groups in methods:
        if name == "none":
            calibrated_pred = pred_raw
            biases: dict[str, float] = {}
        else:
            calibrated_pred, biases = _apply_log_bias_calibration(
                target_raw=target_raw,
                pred_log=pred_log,
                target_log=target_log,
                group_ids=groups,
            )
        summary = _summary_from_arrays(target_raw, calibrated_pred, target_log, np.log1p(calibrated_pred))
        rows.append(
            {
                "method": name,
                "within25_ratio": summary["within25_ratio"],
                "relative_mae": summary["relative_mae"],
                "symmetric_relative_mae": summary["symmetric_relative_mae"],
                "log_mae": summary["log_mae"],
                "pred_target_ratio": summary["pred_target_ratio"],
                "bias_count": len(biases),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
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
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"within25_diagnostics_{args.split}")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.within25_diagnostics", log_file="within25_diagnostics.log")
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
    logger.info("Running within25 diagnostics | split=%s | cases=%s | samples=%s | device=%s", args.split, len(case_dirs), len(sample_paths), device)

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

    frequency_labels = _labels(FREQUENCY_BINS)
    point_target_labels = _labels(POINT_TARGET_BINS)
    sample_mean_labels = _labels(SAMPLE_TARGET_MEAN_BINS)
    bucket_groups: dict[str, dict[str, BucketStats]] = {
        "frequency": {},
        "point_target": {},
        "sample_target_mean": {},
        "region": {},
        "mode_proximity": {},
    }
    case_region_cache: dict[str, dict[str, np.ndarray]] = {}
    case_mode_cache: dict[str, np.ndarray] = {}

    target_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    target_log_parts: list[np.ndarray] = []
    pred_log_parts: list[np.ndarray] = []
    frequency_id_parts: list[np.ndarray] = []
    point_target_id_parts: list[np.ndarray] = []
    sample_mean_id_parts: list[np.ndarray] = []
    region_id_parts: list[np.ndarray] = []
    mode_proximity_id_parts: list[np.ndarray] = []

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
            node_rows = host_batch.node_indices.numpy().astype(np.int64, copy=False)

            frequency_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            sample_mean_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            region_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            mode_proximity_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            for sample_idx, case_name in enumerate(host_batch.case_names):
                sample_mask = (host_batch.sample_index.numpy() == sample_idx)
                sample_frequency = float(host_batch.frequency_hz[sample_idx].item())
                frequency_ids[sample_mask] = _bin_indices(np.array([sample_frequency], dtype=np.float32), FREQUENCY_BINS)[0]
                sample_mean = float(target_raw[sample_mask].mean()) if sample_mask.any() else 0.0
                sample_mean_ids[sample_mask] = _bin_indices(np.array([sample_mean], dtype=np.float32), SAMPLE_TARGET_MEAN_BINS)[0]
                if case_name not in case_region_cache:
                    case_region_cache[case_name] = _case_region_arrays(case_index[case_name])
                region_ids[sample_mask] = _exclusive_region_ids(case_region_cache[case_name], node_rows[sample_mask])
                if case_name not in case_mode_cache:
                    case_mode_cache[case_name] = _case_mode_frequencies(case_index[case_name])
                mode_proximity_ids[sample_mask] = _mode_proximity_id(case_mode_cache[case_name], sample_frequency)

            point_target_ids = _bin_indices(target_raw, POINT_TARGET_BINS).astype(np.int16, copy=False)

            _update_bucket_group(bucket_groups["frequency"], "frequency_bucket", frequency_labels, frequency_ids, target_raw, pred_raw, target_log, pred_log)
            _update_bucket_group(bucket_groups["point_target"], "point_target_bucket", point_target_labels, point_target_ids, target_raw, pred_raw, target_log, pred_log)
            _update_bucket_group(bucket_groups["sample_target_mean"], "sample_target_mean_bucket", sample_mean_labels, sample_mean_ids, target_raw, pred_raw, target_log, pred_log)
            _update_bucket_group(bucket_groups["region"], "region_bucket", REGION_LABELS, region_ids, target_raw, pred_raw, target_log, pred_log)
            _update_bucket_group(
                bucket_groups["mode_proximity"],
                "mode_proximity_bucket",
                MODE_PROXIMITY_LABELS,
                mode_proximity_ids,
                target_raw,
                pred_raw,
                target_log,
                pred_log,
            )

            target_parts.append(target_raw.copy())
            pred_parts.append(pred_raw.copy())
            target_log_parts.append(target_log.copy())
            pred_log_parts.append(pred_log.copy())
            frequency_id_parts.append(frequency_ids.copy())
            point_target_id_parts.append(point_target_ids.copy())
            sample_mean_id_parts.append(sample_mean_ids.copy())
            region_id_parts.append(region_ids.copy())
            mode_proximity_id_parts.append(mode_proximity_ids.copy())

            if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
                logger.info("progress | sample_batch=%s/%s | collected_points=%s", loader_step, len(loader), sum(part.size for part in target_parts))

    target_all = np.concatenate(target_parts)
    pred_all = np.concatenate(pred_parts)
    target_log_all = np.concatenate(target_log_parts)
    pred_log_all = np.concatenate(pred_log_parts)
    frequency_ids_all = np.concatenate(frequency_id_parts)
    point_target_ids_all = np.concatenate(point_target_id_parts)
    sample_mean_ids_all = np.concatenate(sample_mean_id_parts)
    region_ids_all = np.concatenate(region_id_parts)
    mode_proximity_ids_all = np.concatenate(mode_proximity_id_parts)

    overall = _summary_from_arrays(target_all, pred_all, target_log_all, pred_log_all)
    write_json(output_dir / "overall_summary.json", overall)

    outputs: dict[str, str] = {}
    for name, buckets in bucket_groups.items():
        rows = [stats.row(*key.split(":", 1)) for key, stats in buckets.items()]
        if name == "frequency":
            order = {label: idx for idx, label in enumerate(frequency_labels)}
        elif name == "point_target":
            order = {label: idx for idx, label in enumerate(point_target_labels)}
        elif name == "sample_target_mean":
            order = {label: idx for idx, label in enumerate(sample_mean_labels)}
        elif name == "region":
            order = {label: idx for idx, label in enumerate(REGION_LABELS)}
        else:
            order = {label: idx for idx, label in enumerate(MODE_PROXIMITY_LABELS)}
        rows.sort(key=lambda row: order.get(str(row["bucket"]), 999))
        path = output_dir / f"{name}_buckets.csv"
        _write_bucket_csv(path, rows)
        outputs[name] = str(path)
        plot_path = output_dir / f"{name}_within25.png"
        _plot_bucket_table(plot_path, path, title=f"{name} buckets")
        outputs[f"{name}_plot"] = str(plot_path)

    cross_specs = [
        (
            "frequency_x_point_target",
            "frequency",
            frequency_labels,
            frequency_ids_all,
            "point_target",
            point_target_labels,
            point_target_ids_all,
        ),
        (
            "mode_proximity_x_point_target",
            "mode_proximity",
            MODE_PROXIMITY_LABELS,
            mode_proximity_ids_all,
            "point_target",
            point_target_labels,
            point_target_ids_all,
        ),
        (
            "region_x_point_target",
            "region",
            REGION_LABELS,
            region_ids_all,
            "point_target",
            point_target_labels,
            point_target_ids_all,
        ),
    ]
    for name, row_type, row_labels, row_ids, col_type, col_labels, col_ids in cross_specs:
        rows = _cross_bucket_rows(
            row_type=row_type,
            row_labels=row_labels,
            row_ids=row_ids,
            col_type=col_type,
            col_labels=col_labels,
            col_ids=col_ids,
            target_raw=target_all,
            pred_raw=pred_all,
            target_log=target_log_all,
            pred_log=pred_log_all,
        )
        path = output_dir / f"{name}.csv"
        _write_cross_bucket_csv(path, rows)
        outputs[name] = str(path)
        plot_path = output_dir / f"{name}_heatmap.png"
        _plot_cross_bucket_heatmap(plot_path, path, title=name)
        outputs[f"{name}_plot"] = str(plot_path)

    calibration_rows = _calibration_rows(
        target_raw=target_all,
        pred_raw=pred_all,
        pred_log=pred_log_all,
        target_log=target_log_all,
        frequency_ids=frequency_ids_all,
        point_target_ids=point_target_ids_all,
        sample_mean_ids=sample_mean_ids_all,
        region_ids=region_ids_all,
    )
    calibration_path = output_dir / "calibration_oracle.csv"
    with calibration_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(calibration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibration_rows)
    outputs["calibration_oracle"] = str(calibration_path)
    calibration_plot = output_dir / "calibration_oracle.png"
    _plot_calibration(calibration_plot, calibration_rows)
    outputs["calibration_oracle_plot"] = str(calibration_plot)

    error_distribution_plot = output_dir / "error_distribution.png"
    error_cdf_plot = output_dir / "relative_error_cdf.png"
    _plot_error_distribution(error_distribution_plot, target_all, pred_all, target_log_all, pred_log_all)
    _plot_error_cdf(error_cdf_plot, target_all, pred_all)
    outputs["error_distribution_plot"] = str(error_distribution_plot)
    outputs["relative_error_cdf_plot"] = str(error_cdf_plot)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "overall": overall,
        "outputs": outputs,
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

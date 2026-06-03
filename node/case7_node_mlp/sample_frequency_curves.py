from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from case7_node_mlp.data import (
    PER_FREQUENCY_TARGET_COLUMN,
    _load_aligned_target_column,
    _load_case_static,
    build_node_selection_mask,
    discover_case_index,
    expand_case_sample_paths,
    load_raw_point_sample,
    resolve_case_splits,
)
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.trainer import _apply_target_floor, _decode_prediction, _resolve_threshold_from_config, prepare_point_sample
from case7_node_mlp.trace_node_frequency_response import (
    _load_model,
    _mode_frequencies,
    _node_label,
    _predict_sample,
)


TRACE_FIELDNAMES = [
    "split",
    "case_name",
    "node_type",
    "node_rank",
    "node_index",
    "node_label",
    "frequency_hz",
    "target_raw",
    "pred_raw",
    "target_log",
    "pred_log",
    "abs_error",
    "relative_error",
    "symmetric_relative_error",
    "log_abs_error",
    "within25_hit",
    "nearest_mode_frequency_hz",
    "nearest_mode_delta_hz",
    "nearest_mode_relative_delta",
    "is_near_mode_2pct",
    "is_near_mode_5pct",
]

SUMMARY_FIELDNAMES = [
    "split",
    "case_name",
    "node_type",
    "node_rank",
    "node_index",
    "node_label",
    "points",
    "target_mean",
    "pred_mean",
    "pred_target_ratio",
    "mae",
    "rmse",
    "log_mae",
    "log_rmse",
    "relative_mae",
    "symmetric_relative_mae",
    "within25_ratio",
    "under_pred_ratio",
    "over_pred_ratio",
    "target_peak",
    "pred_peak",
    "peak_relative_error",
    "target_peak_frequency_hz",
    "pred_peak_frequency_hz",
    "peak_frequency_delta_hz",
    "freq_delta_log_mae",
    "freq_delta_raw_mae",
    "log_curve_corr",
    "plot",
    "curve_plot",
    "response_bin_plot",
]

CASE_RESPONSE_BUCKET_FIELDNAMES = [
    "split",
    "case_name",
    "target_bucket",
    "bucket_order",
    "points",
    "point_fraction",
    "target_min",
    "target_max",
    "target_mean",
    "pred_mean",
    "pred_target_ratio",
    "log_mae",
    "mae",
    "relative_mae",
    "symmetric_relative_mae",
    "within25_ratio",
    "under_pred_ratio",
    "over_pred_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample hotspot/background nodes and compare full-frequency prediction curves.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a node MLP checkpoint, usually best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--background-quantile",
        type=float,
        default=0.10,
        help="Pick non-hot nodes from the bottom quantile by max/mean response.",
    )
    parser.add_argument(
        "--respect-selection",
        action="store_true",
        help="Only trace frequencies where the node survives dataset selection. Default forces the fixed node at every frequency.",
    )
    parser.add_argument("--case-response-point-batch-size", type=int, default=1048576)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _target_floor_from_config(target_cfg: dict[str, Any], feature_schema: dict[str, Any]) -> float:
    return float(_resolve_threshold_from_config(target_cfg, feature_schema, "zero_below"))


def _load_case_targets_fast(
    case_dir: Path,
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    target_floor: float,
) -> pd.DataFrame:
    nodes_df, _payload, _earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    rows: list[pd.DataFrame] = []
    for sample_path in sample_paths:
        target_df = pd.read_csv(
            sample_path,
            usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN},
        )
        target_values = _load_aligned_target_column(
            target_df,
            target_column=PER_FREQUENCY_TARGET_COLUMN,
            nodes_df=nodes_df,
            target_path=sample_path,
        )
        selected_mask = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg, target_values=target_values)
        selected_indices = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
        if selected_indices.size == 0:
            continue
        selected_targets = _apply_target_floor(
            torch.tensor(target_values[selected_indices], dtype=torch.float32),
            target_floor,
        ).numpy()
        rows.append(
            pd.DataFrame(
                {
                    "frequency_hz": float(sample_path.stem.rsplit("_", 1)[-1].removesuffix("Hz")),
                    "node_index": selected_indices,
                    "target_raw": selected_targets.astype(np.float32, copy=False),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["frequency_hz", "node_index", "target_raw"])
    return pd.concat(rows, ignore_index=True)


def _select_case_nodes(
    targets: pd.DataFrame,
    sample_count: int,
    *,
    background_quantile: float,
    rng: np.random.Generator,
) -> list[tuple[str, int, int]]:
    if targets.empty:
        raise RuntimeError("No valid target rows were found for node selection.")
    stats = (
        targets.groupby("node_index", sort=False)["target_raw"]
        .agg(["count", "mean", "std", "max"])
        .fillna({"std": 0.0})
    )
    min_count = max(3, int(math.ceil(0.80 * max(sample_count, 1))))
    eligible = stats[stats["count"] >= min_count]
    if eligible.empty:
        eligible = stats

    hotspot_nodes = [int(node_index) for node_index in eligible["max"].sort_values(ascending=False).head(2).index]

    background_quantile = min(max(float(background_quantile), 0.01), 0.50)
    max_cutoff = float(eligible["max"].quantile(background_quantile))
    mean_cutoff = float(eligible["mean"].quantile(background_quantile))
    background_candidates = eligible[
        (~eligible.index.isin(hotspot_nodes))
        & (eligible["max"] <= max_cutoff)
        & (eligible["mean"] <= mean_cutoff)
    ]
    if background_candidates.empty:
        background_candidates = eligible[~eligible.index.isin(hotspot_nodes)].copy()
    if background_candidates.empty:
        background_nodes = hotspot_nodes[:2]
    else:
        score = background_candidates["max"].rank(pct=True) + background_candidates["mean"].rank(pct=True)
        lowest = np.asarray(score[score <= float(score.quantile(0.15))].index, dtype=np.int64)
        if lowest.size >= 2:
            background_nodes = [int(value) for value in rng.choice(lowest, size=2, replace=False)]
        else:
            background_nodes = [int(node_index) for node_index in score.sort_values().head(2).index]
    selected: list[tuple[str, int, int]] = []
    for rank, node_index in enumerate(hotspot_nodes, start=1):
        selected.append(("hotspot", rank, int(node_index)))
    for rank, node_index in enumerate(background_nodes, start=1):
        selected.append(("background", rank, int(node_index)))
    return selected


def _add_mode_context(row: dict[str, float], mode_frequencies: np.ndarray) -> dict[str, float]:
    frequency = float(row["frequency_hz"])
    if mode_frequencies.size:
        nearest_idx = int(np.argmin(np.abs(mode_frequencies - frequency)))
        nearest_frequency = float(mode_frequencies[nearest_idx])
        nearest_delta = abs(nearest_frequency - frequency)
    else:
        nearest_frequency = float("nan")
        nearest_delta = float("nan")
    relative_delta = nearest_delta / max(frequency, 1e-12)
    row.update(
        {
            "nearest_mode_frequency_hz": nearest_frequency,
            "nearest_mode_delta_hz": nearest_delta,
            "nearest_mode_relative_delta": relative_delta,
            "is_near_mode_2pct": float(relative_delta <= 0.02),
            "is_near_mode_5pct": float(relative_delta <= 0.05),
        }
    )
    return row


def _trace_node(
    *,
    split: str,
    case_name: str,
    case_dir: Path,
    node_type: str,
    node_rank: int,
    node_index: int,
    sample_paths: list[Path],
    mode_frequencies: np.ndarray,
    model: Any,
    x_scaler: Any,
    y_scaler: Any,
    feature_schema: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    device: torch.device,
    respect_selection: bool,
) -> list[dict[str, Any]]:
    label = _node_label(case_dir, node_index)
    rows: list[dict[str, Any]] = []
    for sample_path in sample_paths:
        row = _predict_sample(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            sample_path=sample_path,
            node_index=node_index,
            device=device,
            respect_selection=respect_selection,
        )
        if row is None:
            continue
        row = _add_mode_context(row, mode_frequencies)
        row.update(
            {
                "split": split,
                "case_name": case_name,
                "node_type": node_type,
                "node_rank": int(node_rank),
                "node_index": int(node_index),
                "node_label": int(label),
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: float(item["frequency_hz"]))
    return rows


def _nanmean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def _summarize_curve(
    rows: list[dict[str, Any]],
    *,
    plot_path: Path,
    response_bin_plot_path: Path,
) -> dict[str, Any]:
    df = pd.DataFrame(rows).sort_values("frequency_hz")
    target_raw = df["target_raw"].to_numpy(dtype=np.float64)
    pred_raw = df["pred_raw"].to_numpy(dtype=np.float64)
    target_log = df["target_log"].to_numpy(dtype=np.float64)
    pred_log = df["pred_log"].to_numpy(dtype=np.float64)
    delta_raw = pred_raw - target_raw
    delta_log = pred_log - target_log
    abs_error = np.abs(delta_raw)
    log_abs_error = np.abs(delta_log)
    relative_mask = np.abs(target_raw) > 1e-12
    relative_error = np.full_like(abs_error, np.nan, dtype=np.float64)
    relative_error[relative_mask] = abs_error[relative_mask] / np.abs(target_raw[relative_mask])

    target_peak_index = int(np.argmax(target_raw)) if target_raw.size else 0
    pred_peak_index = int(np.argmax(pred_raw)) if pred_raw.size else 0
    target_peak = float(target_raw[target_peak_index]) if target_raw.size else float("nan")
    pred_peak = float(pred_raw[pred_peak_index]) if pred_raw.size else float("nan")
    target_mean = float(np.mean(target_raw)) if target_raw.size else float("nan")
    pred_mean = float(np.mean(pred_raw)) if pred_raw.size else float("nan")
    if target_log.size >= 2:
        freq_delta_log_mae = float(np.mean(np.abs(np.diff(pred_log) - np.diff(target_log))))
        freq_delta_raw_mae = float(np.mean(np.abs(np.diff(pred_raw) - np.diff(target_raw))))
    else:
        freq_delta_log_mae = float("nan")
        freq_delta_raw_mae = float("nan")
    if target_log.size >= 2 and np.std(target_log) > 1e-12 and np.std(pred_log) > 1e-12:
        log_curve_corr = float(np.corrcoef(target_log, pred_log)[0, 1])
    else:
        log_curve_corr = float("nan")

    within25 = df.loc[relative_mask, "within25_hit"].astype(float)
    return {
        "split": rows[0]["split"],
        "case_name": rows[0]["case_name"],
        "node_type": rows[0]["node_type"],
        "node_rank": int(rows[0]["node_rank"]),
        "node_index": int(rows[0]["node_index"]),
        "node_label": int(rows[0]["node_label"]),
        "points": int(len(df)),
        "target_mean": target_mean,
        "pred_mean": pred_mean,
        "pred_target_ratio": pred_mean / max(target_mean, 1e-12),
        "mae": float(np.mean(abs_error)) if abs_error.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(delta_raw)))) if delta_raw.size else float("nan"),
        "log_mae": float(np.mean(log_abs_error)) if log_abs_error.size else float("nan"),
        "log_rmse": float(np.sqrt(np.mean(np.square(delta_log)))) if delta_log.size else float("nan"),
        "relative_mae": _nanmean(relative_error),
        "symmetric_relative_mae": float(df["symmetric_relative_error"].mean()),
        "within25_ratio": float(within25.mean()) if not within25.empty else float("nan"),
        "under_pred_ratio": float(np.mean(pred_raw < target_raw)) if target_raw.size else float("nan"),
        "over_pred_ratio": float(np.mean(pred_raw > target_raw)) if target_raw.size else float("nan"),
        "target_peak": target_peak,
        "pred_peak": pred_peak,
        "peak_relative_error": abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12),
        "target_peak_frequency_hz": float(df.iloc[target_peak_index]["frequency_hz"]) if len(df) else float("nan"),
        "pred_peak_frequency_hz": float(df.iloc[pred_peak_index]["frequency_hz"]) if len(df) else float("nan"),
        "peak_frequency_delta_hz": abs(
            float(df.iloc[pred_peak_index]["frequency_hz"]) - float(df.iloc[target_peak_index]["frequency_hz"])
        )
        if len(df)
        else float("nan"),
        "freq_delta_log_mae": freq_delta_log_mae,
        "freq_delta_raw_mae": freq_delta_raw_mae,
        "log_curve_corr": log_curve_corr,
        "plot": str(plot_path),
        "curve_plot": str(plot_path),
        "response_bin_plot": str(response_bin_plot_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _within25_ratio(df: pd.DataFrame) -> float:
    if "within25_hit" not in df.columns:
        return float("nan")
    values = pd.to_numeric(df["within25_hit"], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _within25_text(df: pd.DataFrame) -> str:
    ratio = _within25_ratio(df)
    if not np.isfinite(ratio):
        return "within25=n/a"
    return f"within25={ratio * 100.0:.1f}%"


def _plot_selected_node_curves(path: Path, case_name: str, curves: dict[str, list[dict[str, Any]]]) -> None:
    ordered_keys = [
        "hotspot_1",
        "hotspot_2",
        "background_1",
        "background_2",
    ]
    present_keys = [key for key in ordered_keys if key in curves]
    if not present_keys:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    flat_axes = list(axes.reshape(-1))
    for axis, key in zip(flat_axes, present_keys):
        df = pd.DataFrame(curves[key]).sort_values("frequency_hz")
        node_type = str(df.iloc[0]["node_type"])
        node_rank = int(df.iloc[0]["node_rank"])
        node_index = int(df.iloc[0]["node_index"])
        node_label = int(df.iloc[0]["node_label"])
        target_peak_idx = int(df["target_raw"].idxmax())
        pred_peak_idx = int(df["pred_raw"].idxmax())
        target_peak_freq = float(df.loc[target_peak_idx, "frequency_hz"])
        pred_peak_freq = float(df.loc[pred_peak_idx, "frequency_hz"])
        axis.plot(df["frequency_hz"], df["target_raw"].clip(lower=1e-12), label="target", linewidth=1.8)
        axis.plot(df["frequency_hz"], df["pred_raw"].clip(lower=1e-12), label="prediction", linewidth=1.6)
        axis.scatter(
            [target_peak_freq],
            [max(float(df.loc[target_peak_idx, "target_raw"]), 1e-12)],
            s=28,
            color="tab:blue",
            label=f"target peak {target_peak_freq:.1f}Hz",
        )
        axis.scatter(
            [pred_peak_freq],
            [max(float(df.loc[pred_peak_idx, "pred_raw"]), 1e-12)],
            s=28,
            color="tab:orange",
            label=f"pred peak {pred_peak_freq:.1f}Hz",
        )
        axis.set_yscale("log")
        axis.set_title(f"{node_type}{node_rank} node={node_index} label={node_label} | {_within25_text(df)}")
        axis.set_ylabel("MISES_psd_density")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    for axis in flat_axes[len(present_keys) :]:
        axis.axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel("frequency Hz")
    fig.suptitle(f"{case_name} | selected hotspot/background frequency curves")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _bucket_case_response_rows(
    df: pd.DataFrame,
    *,
    split: str,
    case_name: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    buckets = [
        ("[0,1)", 0.0, 1.0),
        ("[1,10)", 1.0, 10.0),
        ("[10,100)", 10.0, 100.0),
        ("[100,1e3)", 100.0, 1.0e3),
        ("[1e3,1e4)", 1.0e3, 1.0e4),
        ("[1e4,1e5)", 1.0e4, 1.0e5),
        ("[1e5,1e6)", 1.0e5, 1.0e6),
        ("[1e6,1e7)", 1.0e6, 1.0e7),
        ("[1e7,1e8)", 1.0e7, 1.0e8),
        ("[1e8,inf)", 1.0e8, float("inf")),
    ]
    total_points = max(int(len(df)), 1)
    rows: list[dict[str, Any]] = []
    for bucket_order, (label, low, high) in enumerate(buckets):
        mask = df["target_raw"] >= low
        if np.isfinite(high):
            mask &= df["target_raw"] < high
        bucket = df.loc[mask]
        if bucket.empty:
            continue
        target = bucket["target_raw"].to_numpy(dtype=np.float64)
        pred = bucket["pred_raw"].to_numpy(dtype=np.float64)
        abs_error = np.abs(pred - target)
        log_abs_error = np.abs(bucket["pred_log"].to_numpy(dtype=np.float64) - bucket["target_log"].to_numpy(dtype=np.float64))
        relative = bucket["relative_error"].dropna().astype(float)
        symmetric = bucket["symmetric_relative_error"].astype(float)
        within25 = bucket["within25_hit"].dropna().astype(float)
        target_mean = float(np.mean(target))
        pred_mean = float(np.mean(pred))
        rows.append(
            {
                "split": split,
                "case_name": case_name,
                "target_bucket": label,
                "bucket_order": bucket_order,
                "points": int(len(bucket)),
                "point_fraction": float(len(bucket) / total_points),
                "target_min": float(np.min(target)),
                "target_max": float(np.max(target)),
                "target_mean": target_mean,
                "pred_mean": pred_mean,
                "pred_target_ratio": pred_mean / max(target_mean, 1e-12),
                "log_mae": float(np.mean(log_abs_error)),
                "mae": float(np.mean(abs_error)),
                "relative_mae": float(relative.mean()) if not relative.empty else float("nan"),
                "symmetric_relative_mae": float(symmetric.mean()),
                "within25_ratio": float(within25.mean()) if not within25.empty else float("nan"),
                "under_pred_ratio": float(np.mean(pred < target)),
                "over_pred_ratio": float(np.mean(pred > target)),
            }
        )
    return rows


def _predict_case_response_points(
    *,
    split: str,
    case_name: str,
    sample_paths: list[Path],
    model: Any,
    x_scaler: Any,
    y_scaler: Any,
    feature_schema: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    device: torch.device,
    point_batch_size: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    point_batch_size = max(1, int(point_batch_size))
    for sample_path in sample_paths:
        raw = load_raw_point_sample(sample_path, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
        prepared = prepare_point_sample(
            raw,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
        )
        pred_log_chunks: list[torch.Tensor] = []
        pred_raw_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, prepared.num_points, point_batch_size):
                stop = min(start + point_batch_size, prepared.num_points)
                prediction_scaled = model(prepared.features[start:stop].to(device))
                pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled, y_scaler)
                pred_log_chunks.append(pred_log_t.detach().cpu().reshape(-1))
                pred_raw_chunks.append(pred_raw_t.detach().cpu().reshape(-1))
        pred_log = torch.cat(pred_log_chunks).numpy()
        pred_raw = torch.cat(pred_raw_chunks).numpy()
        target_raw = prepared.target_raw.numpy()
        target_log = prepared.target_log.reshape(-1).numpy()
        abs_error = np.abs(pred_raw - target_raw)
        relative_error = np.full_like(abs_error, np.nan, dtype=np.float32)
        relative_mask = np.abs(target_raw) > 1e-12
        relative_error[relative_mask] = abs_error[relative_mask] / np.abs(target_raw[relative_mask])
        symmetric_relative_error = abs_error / np.maximum(0.5 * (np.abs(pred_raw) + np.abs(target_raw)), 1e-12)
        within25 = np.full_like(abs_error, np.nan, dtype=np.float32)
        within25[relative_mask] = (relative_error[relative_mask] <= 0.25).astype(np.float32)
        rows.append(
            pd.DataFrame(
                {
                    "split": split,
                    "case_name": case_name,
                    "sample_name": prepared.name,
                    "frequency_hz": float(prepared.frequency_hz),
                    "node_index": prepared.node_indices.numpy().astype(np.int64, copy=False),
                    "target_raw": target_raw.astype(np.float32, copy=False),
                    "pred_raw": pred_raw.astype(np.float32, copy=False),
                    "target_log": target_log.astype(np.float32, copy=False),
                    "pred_log": pred_log.astype(np.float32, copy=False),
                    "abs_error": abs_error.astype(np.float32, copy=False),
                    "relative_error": relative_error.astype(np.float32, copy=False),
                    "symmetric_relative_error": symmetric_relative_error.astype(np.float32, copy=False),
                    "within25_hit": within25.astype(np.float32, copy=False),
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_case_response_buckets(path: Path, case_name: str, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows).sort_values("bucket_order")
    if df.empty:
        return
    labels = df["target_bucket"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x, df["within25_ratio"].astype(float), color="tab:blue", alpha=0.82, label="within25 ratio")
    axes[0].axhline(0.25, color="tab:red", linestyle="--", linewidth=1.1, label="25% reference")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("within25 ratio")
    axes[0].set_title(f"{case_name} | within25 by target response bucket")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    for idx, value in enumerate(df["within25_ratio"].astype(float)):
        if np.isfinite(value):
            axes[0].text(idx, min(value + 0.03, 0.98), f"{value * 100.0:.0f}%", ha="center", va="bottom", fontsize=8)

    point_fraction = df["point_fraction"].astype(float)
    point_counts = df["points"].astype(int)
    axes[1].bar(x, point_fraction, color="tab:gray", alpha=0.65, label="bucket point share")
    ratio = df["pred_target_ratio"].astype(float).replace([np.inf, -np.inf], np.nan)
    axes_ratio = axes[1].twinx()
    axes_ratio.plot(x, ratio, color="tab:orange", marker="o", linewidth=1.6, label="pred/target mean")
    axes_ratio.axhline(1.0, color="tab:green", linestyle="--", linewidth=1.0, label="ratio=1")
    axes[1].set_ylim(0.0, max(float(point_fraction.max()) * 1.25, 0.05))
    axes[1].set_ylabel("bucket point share")
    axes_ratio.set_ylabel("pred/target mean")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, (fraction, count) in enumerate(zip(point_fraction, point_counts)):
        axes[1].text(
            idx,
            min(float(fraction) + 0.01, axes[1].get_ylim()[1] * 0.96),
            f"{fraction * 100.0:.1f}%\n{count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    lines, line_labels = axes[1].get_legend_handles_labels()
    ratio_lines, ratio_labels = axes_ratio.get_legend_handles_labels()
    axes[1].legend(lines + ratio_lines, line_labels + ratio_labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _summarize_by_type(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary_rows:
        return []
    df = pd.DataFrame(summary_rows)
    metric_columns = [
        "points",
        "target_mean",
        "pred_mean",
        "pred_target_ratio",
        "mae",
        "rmse",
        "log_mae",
        "log_rmse",
        "relative_mae",
        "symmetric_relative_mae",
        "within25_ratio",
        "under_pred_ratio",
        "over_pred_ratio",
        "target_peak",
        "pred_peak",
        "peak_relative_error",
        "peak_frequency_delta_hz",
        "freq_delta_log_mae",
        "freq_delta_raw_mae",
        "log_curve_corr",
    ]
    rows: list[dict[str, Any]] = []
    for node_type, group in df.groupby("node_type", sort=False):
        row: dict[str, Any] = {"node_type": node_type, "curves": int(len(group))}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
        rows.append(row)
    return rows


def run_frequency_curve_diagnostics(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    split: str,
    output_dir: Path,
    num_cases: int,
    seed: int,
    device: torch.device | str,
    checkpoint: dict[str, Any] | None = None,
    model: Any | None = None,
    x_scaler: Any | None = None,
    y_scaler: Any | None = None,
    feature_schema: dict[str, Any] | None = None,
    dataset_cfg: dict[str, Any] | None = None,
    feature_cfg: dict[str, Any] | None = None,
    target_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
    background_quantile: float = 0.10,
    respect_selection: bool = False,
    case_response_point_batch_size: int = 1048576,
    logger: Any | None = None,
) -> dict[str, Any]:
    if int(num_cases) <= 0:
        raise ValueError("num_cases must be positive for frequency curve diagnostics.")
    checkpoint_path = Path(checkpoint_path)
    output_dir = ensure_dir(output_dir)
    device = device if isinstance(device, torch.device) else resolve_device(str(device))
    dataset_cfg = dict(dataset_cfg or config["dataset"])
    feature_cfg = dict(feature_cfg or config.get("features", {}))
    target_cfg = dict(target_cfg or config.get("target", {}))
    loss_cfg = dict(loss_cfg or config.get("loss", {}))

    if logger is None:
        logger = make_logger(
            output_dir,
            logger_name="case7_node_mlp.sample_frequency_curves",
            log_file="sample_frequency_curves.log",
        )

    if checkpoint is None:
        checkpoint = _load_checkpoint(checkpoint_path)
    if model is None or x_scaler is None or y_scaler is None or feature_schema is None:
        model, x_scaler, y_scaler, feature_schema = _load_model(checkpoint, config, device)
    else:
        model.eval()
    feature_schema = dict(feature_schema)
    target_floor = _target_floor_from_config(target_cfg, feature_schema)
    logger.info("Using target floor %.6g from target.zero_below", target_floor)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    split_cases = list(split_names[split])
    if not split_cases:
        raise RuntimeError(f"Split '{split}' has no cases.")
    rng = np.random.default_rng(int(seed))
    selected_case_count = min(max(int(num_cases), 1), len(split_cases))
    selected_cases = sorted(rng.choice(split_cases, size=selected_case_count, replace=False).tolist())
    logger.info(
        "Sampling frequency curves | split=%s | selected_cases=%s/%s | device=%s",
        split,
        selected_case_count,
        len(split_cases),
        device,
    )

    all_trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_response_bucket_rows: list[dict[str, Any]] = []
    plot_dir = ensure_dir(output_dir / "plots")

    for case_offset, case_name in enumerate(selected_cases, start=1):
        case_dir = case_index[case_name]
        sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
        logger.info("Case %s/%s | %s | frames=%s", case_offset, selected_case_count, case_name, len(sample_paths))
        targets = _load_case_targets_fast(
            case_dir=case_dir,
            sample_paths=sample_paths,
            dataset_cfg=dataset_cfg,
            target_floor=target_floor,
        )
        selected_nodes = _select_case_nodes(
            targets,
            sample_count=len(sample_paths),
            background_quantile=float(background_quantile),
            rng=rng,
        )
        logger.info(
            "Selected nodes | case=%s | %s",
            case_name,
            ", ".join(f"{node_type}{rank}={node_index}" for node_type, rank, node_index in selected_nodes),
        )

        mode_frequencies = _mode_frequencies(case_dir)
        curves: dict[str, list[dict[str, Any]]] = {}
        for node_type, node_rank, node_index in selected_nodes:
            rows = _trace_node(
                split=split,
                case_name=case_name,
                case_dir=case_dir,
                node_type=node_type,
                node_rank=int(node_rank),
                node_index=int(node_index),
                sample_paths=sample_paths,
                mode_frequencies=mode_frequencies,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                feature_schema=feature_schema,
                target_cfg=target_cfg,
                loss_cfg=loss_cfg,
                dataset_cfg=dataset_cfg,
                feature_cfg=feature_cfg,
                device=device,
                respect_selection=bool(respect_selection),
            )
            if not rows:
                logger.warning("No rows traced | case=%s | node_type=%s | rank=%s | node=%s", case_name, node_type, node_rank, node_index)
                continue
            curves[f"{node_type}_{node_rank}"] = rows
            all_trace_rows.extend(rows)

        case_response_df = _predict_case_response_points(
            split=split,
            case_name=case_name,
            sample_paths=sample_paths,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            device=device,
            point_batch_size=int(case_response_point_batch_size),
        )
        case_bucket_rows = _bucket_case_response_rows(case_response_df, split=split, case_name=case_name)
        response_bin_plot_path = plot_dir / f"{case_name}_response_buckets.png"
        _plot_case_response_buckets(response_bin_plot_path, case_name, case_bucket_rows)
        case_response_bucket_rows.extend(case_bucket_rows)

        curve_plot_path = plot_dir / f"{case_name}_selected_nodes_frequency_curves.png"
        _plot_selected_node_curves(curve_plot_path, case_name, curves)
        for rows in curves.values():
            summary_rows.append(
                _summarize_curve(
                    rows,
                    plot_path=curve_plot_path,
                    response_bin_plot_path=response_bin_plot_path,
                )
            )

    trace_path = output_dir / "frequency_curve_points.csv"
    curve_summary_path = output_dir / "curve_summary.csv"
    case_response_bucket_path = output_dir / "case_response_buckets.csv"
    type_summary_path = output_dir / "summary_by_type.csv"
    _write_csv(trace_path, all_trace_rows, TRACE_FIELDNAMES)
    _write_csv(curve_summary_path, summary_rows, SUMMARY_FIELDNAMES)
    _write_csv(case_response_bucket_path, case_response_bucket_rows, CASE_RESPONSE_BUCKET_FIELDNAMES)
    type_rows = _summarize_by_type(summary_rows)
    if type_rows:
        _write_csv(type_summary_path, type_rows, list(type_rows[0].keys()))
    run_summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_cases": selected_case_count,
        "seed": int(seed),
        "cases": selected_cases,
        "trace_csv": str(trace_path),
        "curve_summary_csv": str(curve_summary_path),
        "case_response_buckets_csv": str(case_response_bucket_path),
        "summary_by_type_csv": str(type_summary_path),
        "plot_dir": str(plot_dir),
        "summary_by_type": type_rows,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    logger.info("Wrote trace rows: %s", trace_path)
    logger.info("Wrote curve summary: %s", curve_summary_path)
    logger.info("Wrote case response buckets: %s", case_response_bucket_path)
    logger.info("Wrote type summary: %s", type_summary_path)
    logger.info("Wrote plots: %s", plot_dir)
    return run_summary


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    output_dir = ensure_dir(
        args.output_dir
        or checkpoint_path.parent / f"frequency_curve_diagnostics_{args.split}_{args.num_cases}cases"
    )
    logger = make_logger(
        output_dir,
        logger_name="case7_node_mlp.sample_frequency_curves",
        log_file="sample_frequency_curves.log",
    )
    device = resolve_device(args.device)
    run_frequency_curve_diagnostics(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        split=args.split,
        output_dir=output_dir,
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        device=device,
        background_quantile=float(args.background_quantile),
        respect_selection=bool(args.respect_selection),
        case_response_point_batch_size=int(args.case_response_point_batch_size),
        logger=logger,
    )


if __name__ == "__main__":
    main()

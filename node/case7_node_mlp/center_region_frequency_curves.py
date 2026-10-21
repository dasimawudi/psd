from __future__ import annotations

import argparse
import csv
import json
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
    _frequency_from_path,
    _load_aligned_target_column,
    _load_case_static,
    _node_mask_values,
    build_node_selection_mask,
    discover_case_index,
    expand_case_sample_paths,
    resolve_case_splits,
)
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.trainer import _apply_target_floor, _resolve_threshold_from_config
from case7_node_mlp.trace_node_frequency_response import _load_model, _mode_frequencies, _predict_sample


TRACE_FIELDNAMES = [
    "split",
    "case_name",
    "node_type",
    "node_rank",
    "node_position",
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
    "node_position",
    "node_index",
    "node_label",
    "points",
    "target_mean",
    "pred_mean",
    "pred_target_mean_ratio",
    "mae",
    "rmse",
    "log_mae",
    "log_rmse",
    "relative_mae",
    "symmetric_relative_mae",
    "within25_ratio",
    "near_mode_5pct_within25_ratio",
    "far_mode_5pct_within25_ratio",
    "under_pred_ratio",
    "over_pred_ratio",
    "target_peak",
    "pred_peak",
    "pred_target_peak_ratio",
    "peak_relative_error",
    "target_peak_frequency_hz",
    "pred_peak_frequency_hz",
    "peak_frequency_delta_hz",
    "target_peak_nearest_mode_relative_delta",
    "pred_peak_nearest_mode_relative_delta",
    "freq_delta_log_mae",
    "freq_delta_raw_mae",
    "log_curve_corr",
    "plot",
]

CASE_NODE_FIELDNAMES = [
    "split",
    "case_name",
    "node_type",
    "node_rank",
    "node_position",
    "node_index",
    "node_label",
    "valid_frequency_count",
    "target_mean",
    "target_std",
    "target_peak",
    "target_peak_frequency_hz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace fitted prediction curves for high-response nodes in the disk center region.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a node MLP checkpoint, usually best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-cases", type=int, default=8, help="Number of split cases to sample when --case-name is omitted.")
    parser.add_argument("--case-name", action="append", default=None, help="Specific case to plot. Can be passed more than once.")
    parser.add_argument("--nodes-per-case", type=int, default=4, help="Number of center-region peak nodes traced per case.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional quick-run frame cap.")
    parser.add_argument(
        "--min-frequency-coverage",
        type=float,
        default=0.80,
        help="Minimum fraction of frames with valid targets before a node can be selected.",
    )
    parser.add_argument(
        "--respect-selection",
        action="store_true",
        help="Only trace frequencies where selected nodes survive dataset selection. Default forces the fixed node at every frequency.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _target_floor_from_config(target_cfg: dict[str, Any], feature_schema: dict[str, Any]) -> float:
    return float(_resolve_threshold_from_config(target_cfg, feature_schema, "zero_below"))


def _node_identity(nodes_df: pd.DataFrame, node_position: int) -> tuple[int, int]:
    row = nodes_df.iloc[int(node_position)]
    node_index = int(row["node_index"]) if "node_index" in nodes_df.columns else int(node_position)
    node_label = int(row["node_label"]) if "node_label" in nodes_df.columns else node_index
    return node_index, node_label


def _center_region_mask(nodes_df: pd.DataFrame) -> np.ndarray:
    return _node_mask_values(nodes_df, "center_couple_mask") | _node_mask_values(nodes_df, "center_node_mask")


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


def _load_center_node_stats(
    *,
    split: str,
    case_dir: Path,
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    target_floor: float,
    min_frequency_coverage: float,
) -> pd.DataFrame:
    nodes_df, _payload, _earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    base_selection = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg)
    center_selection = base_selection & _center_region_mask(nodes_df)
    node_positions = np.flatnonzero(center_selection).astype(np.int64, copy=False)
    if node_positions.size == 0:
        return pd.DataFrame(columns=CASE_NODE_FIELDNAMES)

    counts = np.zeros(node_positions.size, dtype=np.int32)
    sums = np.zeros(node_positions.size, dtype=np.float64)
    sumsq = np.zeros(node_positions.size, dtype=np.float64)
    peaks = np.full(node_positions.size, -np.inf, dtype=np.float64)
    peak_frequencies = np.full(node_positions.size, np.nan, dtype=np.float64)

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
        selected_values = _apply_target_floor(
            torch.tensor(target_values[node_positions], dtype=torch.float32),
            target_floor,
        ).numpy()
        valid = np.isfinite(selected_values) & (selected_values >= 0.0)
        if not bool(valid.any()):
            continue
        valid_values = selected_values[valid].astype(np.float64, copy=False)
        counts[valid] += 1
        sums[valid] += valid_values
        sumsq[valid] += np.square(valid_values)
        better_peak = valid & (selected_values > peaks)
        if bool(better_peak.any()):
            peaks[better_peak] = selected_values[better_peak]
            peak_frequencies[better_peak] = float(_frequency_from_path(sample_path))

    min_count = max(1, int(math.ceil(float(min_frequency_coverage) * max(len(sample_paths), 1))))
    eligible = (counts >= min_count) & np.isfinite(peaks) & (peaks >= 0.0)
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(eligible):
        node_position = int(node_positions[int(local_index)])
        node_index, node_label = _node_identity(nodes_df, node_position)
        count = int(counts[int(local_index)])
        mean = float(sums[int(local_index)] / max(count, 1))
        variance = max(float(sumsq[int(local_index)] / max(count, 1) - mean * mean), 0.0)
        rows.append(
            {
                "split": split,
                "case_name": case_dir.name,
                "node_position": node_position,
                "node_index": node_index,
                "node_label": node_label,
                "valid_frequency_count": count,
                "target_mean": mean,
                "target_std": float(math.sqrt(variance)),
                "target_peak": float(peaks[int(local_index)]),
                "target_peak_frequency_hz": float(peak_frequencies[int(local_index)]),
            }
        )
    return pd.DataFrame(rows)


def _select_center_nodes(stats: pd.DataFrame, nodes_per_case: int) -> list[dict[str, Any]]:
    if stats.empty:
        return []
    ordered = stats.sort_values(["target_peak", "target_mean", "node_position"], ascending=[False, False, True])
    selected: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(ordered.head(max(int(nodes_per_case), 1)).iterrows(), start=1):
        selected.append(
            {
                "split": row["split"],
                "case_name": row["case_name"],
                "node_type": "center_peak",
                "node_rank": int(rank),
                "node_position": int(row["node_position"]),
                "node_index": int(row["node_index"]),
                "node_label": int(row["node_label"]),
                "valid_frequency_count": int(row["valid_frequency_count"]),
                "target_mean": float(row["target_mean"]),
                "target_std": float(row["target_std"]),
                "target_peak": float(row["target_peak"]),
                "target_peak_frequency_hz": float(row["target_peak_frequency_hz"]),
            }
        )
    return selected


def _trace_node(
    *,
    split: str,
    case_name: str,
    case_dir: Path,
    selected_node: dict[str, Any],
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
    rows: list[dict[str, Any]] = []
    node_position = int(selected_node["node_position"])
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
            node_index=node_position,
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
                "node_type": selected_node["node_type"],
                "node_rank": int(selected_node["node_rank"]),
                "node_position": node_position,
                "node_index": int(selected_node["node_index"]),
                "node_label": int(selected_node["node_label"]),
            }
        )
        rows.append(row)
    rows.sort(key=lambda item: float(item["frequency_hz"]))
    return rows


def _nanmean(values: np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def _within25_for_mask(df: pd.DataFrame, mask: np.ndarray | pd.Series) -> float:
    values = df.loc[mask, "within25_hit"].dropna().astype(float)
    if values.empty:
        return float("nan")
    return float(values.mean())


def _summarize_curve(rows: list[dict[str, Any]], *, plot_path: Path) -> dict[str, Any]:
    df = pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)
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

    near_mode_mask = df["is_near_mode_5pct"].astype(bool)
    within25 = df.loc[relative_mask, "within25_hit"].dropna().astype(float)
    return {
        "split": rows[0]["split"],
        "case_name": rows[0]["case_name"],
        "node_type": rows[0]["node_type"],
        "node_rank": int(rows[0]["node_rank"]),
        "node_position": int(rows[0]["node_position"]),
        "node_index": int(rows[0]["node_index"]),
        "node_label": int(rows[0]["node_label"]),
        "points": int(len(df)),
        "target_mean": target_mean,
        "pred_mean": pred_mean,
        "pred_target_mean_ratio": pred_mean / max(abs(target_mean), 1e-12),
        "mae": float(np.mean(abs_error)) if abs_error.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(delta_raw)))) if delta_raw.size else float("nan"),
        "log_mae": float(np.mean(log_abs_error)) if log_abs_error.size else float("nan"),
        "log_rmse": float(np.sqrt(np.mean(np.square(delta_log)))) if delta_log.size else float("nan"),
        "relative_mae": _nanmean(relative_error),
        "symmetric_relative_mae": float(df["symmetric_relative_error"].mean()),
        "within25_ratio": float(within25.mean()) if not within25.empty else float("nan"),
        "near_mode_5pct_within25_ratio": _within25_for_mask(df, relative_mask & near_mode_mask.to_numpy()),
        "far_mode_5pct_within25_ratio": _within25_for_mask(df, relative_mask & ~near_mode_mask.to_numpy()),
        "under_pred_ratio": float(np.mean(pred_raw < target_raw)) if target_raw.size else float("nan"),
        "over_pred_ratio": float(np.mean(pred_raw > target_raw)) if target_raw.size else float("nan"),
        "target_peak": target_peak,
        "pred_peak": pred_peak,
        "pred_target_peak_ratio": pred_peak / max(abs(target_peak), 1e-12),
        "peak_relative_error": abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12),
        "target_peak_frequency_hz": float(df.iloc[target_peak_index]["frequency_hz"]) if len(df) else float("nan"),
        "pred_peak_frequency_hz": float(df.iloc[pred_peak_index]["frequency_hz"]) if len(df) else float("nan"),
        "peak_frequency_delta_hz": abs(
            float(df.iloc[pred_peak_index]["frequency_hz"]) - float(df.iloc[target_peak_index]["frequency_hz"])
        )
        if len(df)
        else float("nan"),
        "target_peak_nearest_mode_relative_delta": float(df.iloc[target_peak_index]["nearest_mode_relative_delta"]) if len(df) else float("nan"),
        "pred_peak_nearest_mode_relative_delta": float(df.iloc[pred_peak_index]["nearest_mode_relative_delta"]) if len(df) else float("nan"),
        "freq_delta_log_mae": freq_delta_log_mae,
        "freq_delta_raw_mae": freq_delta_raw_mae,
        "log_curve_corr": log_curve_corr,
        "plot": str(plot_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _within25_text(df: pd.DataFrame) -> str:
    values = df["within25_hit"].dropna().astype(float)
    if values.empty:
        return "within25=n/a"
    return f"within25={float(values.mean()) * 100.0:.1f}%"


def _relative_error_ylim(values: pd.Series) -> tuple[float, float]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if finite.size == 0:
        return 0.0, 1.0
    upper = min(max(1.0, float(np.nanpercentile(finite, 95)) * 1.15), 20.0)
    return 0.0, upper


def _plot_case_curves(path: Path, case_name: str, curves: dict[str, list[dict[str, Any]]]) -> None:
    if not curves:
        return
    ordered_keys = sorted(
        curves,
        key=lambda key: (
            int(pd.DataFrame(curves[key]).iloc[0]["node_rank"]),
            int(pd.DataFrame(curves[key]).iloc[0]["node_position"]),
        ),
    )
    fig, axes = plt.subplots(len(ordered_keys), 2, figsize=(15, max(4.0, 3.6 * len(ordered_keys))), squeeze=False)
    for row_axis, key in enumerate(ordered_keys):
        df = pd.DataFrame(curves[key]).sort_values("frequency_hz")
        first = df.iloc[0]
        target_peak_idx = int(df["target_raw"].idxmax())
        pred_peak_idx = int(df["pred_raw"].idxmax())
        target_peak_freq = float(df.loc[target_peak_idx, "frequency_hz"])
        pred_peak_freq = float(df.loc[pred_peak_idx, "frequency_hz"])
        target_peak = float(df.loc[target_peak_idx, "target_raw"])
        pred_peak = float(df.loc[pred_peak_idx, "pred_raw"])
        peak_ratio = pred_peak / max(abs(target_peak), 1e-12)

        curve_axis = axes[row_axis, 0]
        curve_axis.plot(df["frequency_hz"], df["target_raw"].clip(lower=1e-12), label="target", linewidth=1.8)
        curve_axis.plot(df["frequency_hz"], df["pred_raw"].clip(lower=1e-12), label="prediction", linewidth=1.6)
        curve_axis.scatter([target_peak_freq], [max(target_peak, 1e-12)], s=26, color="tab:blue", label=f"target peak {target_peak_freq:.1f}Hz")
        curve_axis.scatter([pred_peak_freq], [max(pred_peak, 1e-12)], s=26, color="tab:orange", label=f"pred peak {pred_peak_freq:.1f}Hz")
        curve_axis.set_yscale("log")
        curve_axis.set_ylabel(PER_FREQUENCY_TARGET_COLUMN)
        curve_axis.set_title(
            f"rank={int(first['node_rank'])} pos={int(first['node_position'])} "
            f"idx={int(first['node_index'])} label={int(first['node_label'])} | "
            f"peak pred/target={peak_ratio:.3g} | {_within25_text(df)}"
        )
        curve_axis.grid(True, alpha=0.25)
        curve_axis.legend(fontsize=8)

        error_axis = axes[row_axis, 1]
        error_axis.plot(df["frequency_hz"], df["relative_error"], color="tab:red", linewidth=1.5, label="relative error")
        error_axis.axhline(0.25, color="tab:green", linestyle="--", linewidth=1.1, label="25%")
        near_mode = df[df["is_near_mode_5pct"].astype(bool)]
        if not near_mode.empty:
            error_axis.scatter(
                near_mode["frequency_hz"],
                near_mode["relative_error"],
                s=15,
                color="tab:orange",
                alpha=0.8,
                label="near mode <=5%",
            )
        error_axis.set_ylim(*_relative_error_ylim(df["relative_error"]))
        error_axis.set_ylabel("relative error")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend(fontsize=8)

    for axis in axes[-1, :]:
        axis.set_xlabel("frequency Hz")
    fig.suptitle(f"{case_name} | disk center selected-node fitted curves")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_peak_scatter(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return
    target = df["target_peak"].clip(lower=1e-12)
    pred = df["pred_peak"].clip(lower=1e-12)
    lower = float(min(target.min(), pred.min()))
    upper = float(max(target.max(), pred.max()))
    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    scatter = axis.scatter(
        target,
        pred,
        c=df["within25_ratio"].astype(float),
        cmap="viridis",
        s=42,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.25,
    )
    axis.plot([lower, upper], [lower, upper], color="tab:green", linestyle="--", linewidth=1.2, label="pred=target")
    axis.plot([lower, upper], [0.75 * lower, 0.75 * upper], color="tab:gray", linestyle=":", linewidth=1.0, label="25% band")
    axis.plot([lower, upper], [1.25 * lower, 1.25 * upper], color="tab:gray", linestyle=":", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("target peak")
    axis.set_ylabel("predicted peak")
    axis.set_title("Disk center peak fit by traced curve")
    axis.grid(True, alpha=0.25)
    axis.legend()
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("curve within25 ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _summarize_all(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        return {}
    df = pd.DataFrame(summary_rows)
    return {
        "curve_count": int(len(df)),
        "target_peak_median": float(df["target_peak"].median()),
        "pred_peak_median": float(df["pred_peak"].median()),
        "pred_target_peak_ratio_mean": float(df["pred_target_peak_ratio"].mean()),
        "pred_target_peak_ratio_median": float(df["pred_target_peak_ratio"].median()),
        "within25_ratio_mean": float(df["within25_ratio"].mean()),
        "within25_ratio_median": float(df["within25_ratio"].median()),
        "log_curve_corr_mean": float(df["log_curve_corr"].mean()),
        "peak_frequency_delta_hz_median": float(df["peak_frequency_delta_hz"].median()),
        "under_pred_ratio_mean": float(df["under_pred_ratio"].mean()),
    }


def run_center_region_frequency_curves(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    split: str,
    output_dir: Path,
    num_cases: int,
    nodes_per_case: int,
    seed: int,
    device: torch.device | str,
    case_names: list[str] | None = None,
    checkpoint: dict[str, Any] | None = None,
    max_frames_per_case: int | None = None,
    min_frequency_coverage: float = 0.80,
    respect_selection: bool = False,
    logger: Any | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = ensure_dir(output_dir)
    device = device if isinstance(device, torch.device) else resolve_device(str(device))
    dataset_cfg = dict(config["dataset"])
    if max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(max_frames_per_case)
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    if logger is None:
        logger = make_logger(
            output_dir,
            logger_name="case7_node_mlp.center_region_frequency_curves",
            log_file="center_region_frequency_curves.log",
        )

    if checkpoint is None:
        checkpoint = _load_checkpoint(checkpoint_path)
    model, x_scaler, y_scaler, feature_schema = _load_model(checkpoint, config, device)
    feature_schema = dict(feature_schema)
    target_floor = _target_floor_from_config(target_cfg, feature_schema)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    split_case_names = list(split_names[split])
    if case_names:
        selected_cases = [str(name) for name in case_names]
    else:
        rng = np.random.default_rng(int(seed))
        selected_case_count = min(max(int(num_cases), 1), len(split_case_names))
        selected_cases = sorted(rng.choice(split_case_names, size=selected_case_count, replace=False).tolist())
    unknown = [name for name in selected_cases if name not in case_index]
    if unknown:
        raise KeyError(f"Unknown case(s): {unknown}")

    logger.info(
        "Tracing center-region curves | split=%s | cases=%s | nodes_per_case=%s | device=%s | target_floor=%.6g",
        split,
        len(selected_cases),
        int(nodes_per_case),
        device,
        target_floor,
    )

    all_trace_rows: list[dict[str, Any]] = []
    selected_node_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    plot_dir = ensure_dir(output_dir / "plots")

    for case_offset, case_name in enumerate(selected_cases, start=1):
        case_dir = case_index[case_name]
        sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
        logger.info("Case %s/%s | %s | frames=%s", case_offset, len(selected_cases), case_name, len(sample_paths))
        stats = _load_center_node_stats(
            split=split,
            case_dir=case_dir,
            sample_paths=sample_paths,
            dataset_cfg=dataset_cfg,
            target_floor=target_floor,
            min_frequency_coverage=float(min_frequency_coverage),
        )
        selected_nodes = _select_center_nodes(stats, nodes_per_case=int(nodes_per_case))
        if not selected_nodes:
            logger.warning("No eligible center-region nodes | case=%s", case_name)
            continue
        selected_node_rows.extend(selected_nodes)
        logger.info(
            "Selected center nodes | case=%s | %s",
            case_name,
            ", ".join(
                f"rank{row['node_rank']} pos={row['node_position']} peak={row['target_peak']:.4g}@{row['target_peak_frequency_hz']:.1f}Hz"
                for row in selected_nodes
            ),
        )

        mode_frequencies = _mode_frequencies(case_dir)
        curves: dict[str, list[dict[str, Any]]] = {}
        for selected_node in selected_nodes:
            rows = _trace_node(
                split=split,
                case_name=case_name,
                case_dir=case_dir,
                selected_node=selected_node,
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
                logger.warning("No traced rows | case=%s | node_position=%s", case_name, selected_node["node_position"])
                continue
            key = f"center_peak_{int(selected_node['node_rank'])}"
            curves[key] = rows
            all_trace_rows.extend(rows)

        curve_plot_path = plot_dir / f"{case_name}_center_region_curves.png"
        _plot_case_curves(curve_plot_path, case_name, curves)
        for rows in curves.values():
            summary_rows.append(_summarize_curve(rows, plot_path=curve_plot_path))

    trace_path = output_dir / "center_region_curve_points.csv"
    selected_nodes_path = output_dir / "selected_center_nodes.csv"
    curve_summary_path = output_dir / "center_region_curve_summary.csv"
    peak_scatter_path = plot_dir / "center_region_peak_scatter.png"
    _write_csv(trace_path, all_trace_rows, TRACE_FIELDNAMES)
    _write_csv(selected_nodes_path, selected_node_rows, CASE_NODE_FIELDNAMES)
    _write_csv(curve_summary_path, summary_rows, SUMMARY_FIELDNAMES)
    _plot_peak_scatter(peak_scatter_path, summary_rows)

    aggregate = _summarize_all(summary_rows)
    run_summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "seed": int(seed),
        "cases": selected_cases,
        "num_cases": int(len(selected_cases)),
        "nodes_per_case": int(nodes_per_case),
        "target_floor": float(target_floor),
        "respect_selection": bool(respect_selection),
        "trace_csv": str(trace_path),
        "selected_center_nodes_csv": str(selected_nodes_path),
        "curve_summary_csv": str(curve_summary_path),
        "plot_dir": str(plot_dir),
        "peak_scatter_plot": str(peak_scatter_path),
        "aggregate": aggregate,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    logger.info("Wrote trace rows: %s", trace_path)
    logger.info("Wrote selected nodes: %s", selected_nodes_path)
    logger.info("Wrote curve summary: %s", curve_summary_path)
    logger.info("Wrote plots: %s", plot_dir)
    return run_summary


def main() -> None:
    args = parse_args()
    if int(args.nodes_per_case) <= 0:
        raise ValueError("--nodes-per-case must be positive.")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    output_dir = ensure_dir(
        args.output_dir
        or checkpoint_path.parent / f"center_region_frequency_curves_{args.split}_{args.num_cases}cases"
    )
    logger = make_logger(
        output_dir,
        logger_name="case7_node_mlp.center_region_frequency_curves",
        log_file="center_region_frequency_curves.log",
    )
    summary = run_center_region_frequency_curves(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        split=args.split,
        output_dir=output_dir,
        num_cases=int(args.num_cases),
        nodes_per_case=int(args.nodes_per_case),
        seed=int(args.seed),
        device=resolve_device(args.device),
        case_names=args.case_name,
        max_frames_per_case=args.max_frames_per_case,
        min_frequency_coverage=float(args.min_frequency_coverage),
        respect_selection=bool(args.respect_selection),
        logger=logger,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

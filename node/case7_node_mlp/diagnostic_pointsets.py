from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from case7_node_mlp.data import (
    PER_FREQUENCY_DIRNAME,
    PER_FREQUENCY_TARGET_COLUMN,
    STRESS_REGION_GROUPS,
    STRESS_REGION_MASK_COLUMNS,
    _frequency_from_path,
    _load_aligned_target_column,
    _node_mask_values,
    discover_case_index,
    expand_case_sample_paths,
    resolve_case_splits,
)
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, write_json


POINTSET_TYPES = ("full_flat_background", "full_sensitive_flat", "full_hotspot")
REGION_ORDER = (
    "full_part",
    "center_couple_region",
    "plate_hole_region",
    "ear_hole_region",
    "ear_connection_region",
    "other_region",
    "background_non_stress_region",
)


@dataclass
class CaseSpectrum:
    case_name: str
    frequencies: np.ndarray
    nodes_df: pd.DataFrame
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-part diagnostic pointsets for node MLP specialist experiments.")
    parser.add_argument("--config", type=str, required=True, help="Base training config used for dataset root and split.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="node/outputs/diagnostics/fullpart_three_point_specialists",
        help="Output directory for pointsets, summaries and plots.",
    )
    parser.add_argument("--max-cases-per-split", type=int, default=None, help="Optional debug cap per split.")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional debug cap per case.")
    parser.add_argument("--flat-quantile", type=float, default=0.20, help="Flatness quantile for D1/D2.")
    parser.add_argument("--flat-relaxed-quantile", type=float, default=0.30, help="Relaxed flatness quantile.")
    parser.add_argument("--background-distance-quantile", type=float, default=0.60, help="D1 distance quantile.")
    parser.add_argument("--near-region-quantile", type=float, default=0.20, help="D2 near-region distance quantile fallback.")
    parser.add_argument("--hotspot-max-quantile", type=float, default=0.90, help="D3 max response quantile.")
    parser.add_argument("--hotspot-dynamic-quantile", type=float, default=0.80, help="D3 std/peak-ratio quantile.")
    parser.add_argument("--top-fraction", type=float, default=0.05, help="Per-frame top fraction included in hotspot candidates.")
    parser.add_argument("--max-nodes-per-case-pointset", type=int, default=800, help="Cap nodes per case per pointset.")
    parser.add_argument("--write-all-node-stats", action="store_true", help="Write per case-node stats CSV for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_case_names(config: dict[str, Any], max_cases_per_split: int | None) -> dict[str, list[str]]:
    dataset_cfg = dict(config["dataset"])
    splits = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    if max_cases_per_split is not None:
        splits = {split: names[: int(max_cases_per_split)] for split, names in splits.items()}
    return splits


def _frame_paths_for_case(case_dir: Path, dataset_cfg: dict[str, Any], max_frames_per_case: int | None) -> list[Path]:
    local_cfg = dict(dataset_cfg)
    if max_frames_per_case is not None:
        local_cfg["max_frames_per_case"] = int(max_frames_per_case)
    return expand_case_sample_paths([case_dir], local_cfg)


def _load_case_spectrum(case_dir: Path, dataset_cfg: dict[str, Any], max_frames_per_case: int | None) -> CaseSpectrum:
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    frame_paths = _frame_paths_for_case(case_dir, dataset_cfg, max_frames_per_case=max_frames_per_case)
    if not frame_paths:
        raise ValueError(f"No per-frequency frames selected for {case_dir}")
    frequencies: list[float] = []
    values: list[np.ndarray] = []
    for frame_path in frame_paths:
        target_df = pd.read_csv(frame_path, usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN})
        target_values = _load_aligned_target_column(
            target_df=target_df,
            target_column=PER_FREQUENCY_TARGET_COLUMN,
            nodes_df=nodes_df,
            target_path=frame_path,
        )
        frequencies.append(_frequency_from_path(frame_path))
        values.append(target_values.astype(np.float32, copy=False))
    order = np.argsort(np.asarray(frequencies, dtype=np.float32))
    sorted_frequencies = np.asarray(frequencies, dtype=np.float32)[order]
    sorted_values = np.stack(values, axis=0)[order]
    sorted_values = np.where(np.isfinite(sorted_values) & (sorted_values >= 0.0), sorted_values, np.nan)
    return CaseSpectrum(case_name=case_dir.name, frequencies=sorted_frequencies, nodes_df=nodes_df, values=sorted_values)


def _region_masks(nodes_df: pd.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for region_name, columns in STRESS_REGION_GROUPS.items():
        mask = np.zeros(len(nodes_df), dtype=bool)
        for column in columns:
            mask |= _node_mask_values(nodes_df, column)
        masks[region_name] = mask
    stress_mask = np.zeros(len(nodes_df), dtype=bool)
    for column in STRESS_REGION_MASK_COLUMNS:
        stress_mask |= _node_mask_values(nodes_df, column)
    masks["stress_region"] = stress_mask
    masks["background_non_stress_region"] = ~stress_mask
    masks["full_part"] = np.ones(len(nodes_df), dtype=bool)
    masks["other_region"] = ~stress_mask
    return masks


def _region_type_for_nodes(nodes_df: pd.DataFrame) -> np.ndarray:
    result = np.full(len(nodes_df), "other_region", dtype=object)
    for region_name in ("center_couple_region", "plate_hole_region", "ear_hole_region", "ear_connection_region"):
        mask = np.zeros(len(nodes_df), dtype=bool)
        for column in STRESS_REGION_GROUPS[region_name]:
            mask |= _node_mask_values(nodes_df, column)
        result[mask] = region_name
    return result


def _region_distance(nodes_df: pd.DataFrame, region_mask: np.ndarray) -> np.ndarray:
    points = nodes_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    region_points = points[region_mask]
    if region_points.size == 0:
        return np.full(len(nodes_df), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        distance, _ = cKDTree(region_points).query(points, k=1, workers=-1)
        return distance.astype(np.float32, copy=False)
    except Exception:
        output = np.empty(len(points), dtype=np.float32)
        for start in range(0, len(points), 2048):
            chunk = points[start : start + 2048]
            delta = chunk[:, None, :] - region_points[None, :, :]
            output[start : start + len(chunk)] = np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=1))
        return output


def _safe_quantile(values: np.ndarray, quantile: float, default: float = 0.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.quantile(finite, min(max(float(quantile), 0.0), 1.0)))


def _case_node_stats(spectrum: CaseSpectrum, top_fraction: float) -> pd.DataFrame:
    values = spectrum.values
    valid_count = np.isfinite(values).sum(axis=0)
    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    log_values = np.log1p(clean_values)
    mean_log = np.nanmean(log_values, axis=0)
    std_log = np.nanstd(log_values, axis=0)
    range_log = np.nanmax(log_values, axis=0) - np.nanmin(log_values, axis=0)
    mean_raw = np.mean(clean_values, axis=0)
    max_raw = np.max(clean_values, axis=0)
    peak_ratio = max_raw / np.maximum(mean_raw, 1e-12)
    peak_idx = np.argmax(clean_values, axis=0)
    peak_frequency = spectrum.frequencies[peak_idx]

    hotspot_from_frames = np.zeros(values.shape[1], dtype=bool)
    top_count = max(1, int(math.ceil(values.shape[1] * max(float(top_fraction), 0.0))))
    for row in np.nan_to_num(values, nan=-np.inf):
        if top_count >= row.size:
            hotspot_from_frames[:] = True
            continue
        top_indices = np.argpartition(row, -top_count)[-top_count:]
        hotspot_from_frames[top_indices] = True
        hotspot_from_frames[int(np.argmax(row))] = True

    node_index = (
        spectrum.nodes_df["node_index"].to_numpy(dtype=np.int64)
        if "node_index" in spectrum.nodes_df.columns
        else np.arange(len(spectrum.nodes_df), dtype=np.int64)
    )
    region_type = _region_type_for_nodes(spectrum.nodes_df)
    masks = _region_masks(spectrum.nodes_df)
    nearest_stress_distance = np.full(len(spectrum.nodes_df), np.inf, dtype=np.float32)
    for region_name in ("center_couple_region", "plate_hole_region", "ear_hole_region", "ear_connection_region"):
        nearest_stress_distance = np.minimum(nearest_stress_distance, _region_distance(spectrum.nodes_df, masks[region_name]))

    return pd.DataFrame(
        {
            "case_name": spectrum.case_name,
            "node_index": node_index,
            "region_type": region_type,
            "stress_region_mask": masks["stress_region"].astype(np.int8),
            "bc_mask": _node_mask_values(spectrum.nodes_df, "bc_mask").astype(np.int8),
            "center_node_mask": _node_mask_values(spectrum.nodes_df, "center_node_mask").astype(np.int8),
            "valid_frequency_count": valid_count.astype(np.int16),
            "mean_log": mean_log,
            "std_log": std_log,
            "range_log": range_log,
            "mean_raw": mean_raw,
            "max_raw": max_raw,
            "peak_ratio": peak_ratio,
            "peak_frequency": peak_frequency,
            "nearest_stress_distance": nearest_stress_distance,
            "frame_top_candidate": hotspot_from_frames.astype(np.int8),
        }
    )


def _balanced_take(df: pd.DataFrame, max_count: int, seed: int) -> pd.DataFrame:
    if max_count <= 0 or len(df) <= max_count:
        return df
    return df.sample(n=max_count, random_state=seed)


def _select_pointsets_for_split(
    stats_df: pd.DataFrame,
    flat_quantile: float,
    flat_relaxed_quantile: float,
    background_distance_quantile: float,
    near_region_quantile: float,
    hotspot_max_quantile: float,
    hotspot_dynamic_quantile: float,
    max_nodes_per_case_pointset: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    non_bc = (stats_df["bc_mask"] == 0) & (stats_df["center_node_mask"] == 0) & (stats_df["valid_frequency_count"] > 0)

    background_candidates = stats_df[non_bc & (stats_df["stress_region_mask"] == 0)].copy()
    if not background_candidates.empty:
        distance_threshold = _safe_quantile(background_candidates["nearest_stress_distance"].to_numpy(), background_distance_quantile)
        far_background = background_candidates[background_candidates["nearest_stress_distance"] >= distance_threshold]
        base = far_background if len(far_background) else background_candidates
        flat_thresholds = {
            "std_log": _safe_quantile(base["std_log"].to_numpy(), flat_quantile),
            "range_log": _safe_quantile(base["range_log"].to_numpy(), flat_quantile),
            "peak_ratio": _safe_quantile(base["peak_ratio"].to_numpy(), flat_quantile),
        }
        d1 = base[
            (base["std_log"] <= flat_thresholds["std_log"])
            & (base["range_log"] <= flat_thresholds["range_log"])
            & (base["peak_ratio"] <= flat_thresholds["peak_ratio"])
        ].copy()
        if len(d1) < max(100, int(0.01 * len(background_candidates))):
            flat_thresholds = {
                "std_log": _safe_quantile(base["std_log"].to_numpy(), flat_relaxed_quantile),
                "range_log": _safe_quantile(base["range_log"].to_numpy(), flat_relaxed_quantile),
                "peak_ratio": _safe_quantile(base["peak_ratio"].to_numpy(), flat_relaxed_quantile),
            }
            d1 = base[
                (base["std_log"] <= flat_thresholds["std_log"])
                & (base["range_log"] <= flat_thresholds["range_log"])
                & (base["peak_ratio"] <= flat_thresholds["peak_ratio"])
            ].copy()
        if not d1.empty:
            d1["pointset_type"] = "full_flat_background"
            rows.append(_cap_per_case(d1, max_nodes_per_case_pointset, seed))

    sensitive_candidates = stats_df[non_bc & (stats_df["stress_region_mask"] == 1)].copy()
    if not sensitive_candidates.empty:
        d2_parts: list[pd.DataFrame] = []
        for region_name in ("center_couple_region", "plate_hole_region", "ear_hole_region", "ear_connection_region"):
            region = sensitive_candidates[sensitive_candidates["region_type"] == region_name]
            if region.empty:
                continue
            thresholds = {
                "std_log": _safe_quantile(region["std_log"].to_numpy(), flat_quantile),
                "range_log": _safe_quantile(region["range_log"].to_numpy(), flat_quantile),
                "peak_ratio": _safe_quantile(region["peak_ratio"].to_numpy(), flat_quantile),
            }
            selected = region[
                (region["std_log"] <= thresholds["std_log"])
                & (region["range_log"] <= thresholds["range_log"])
                & (region["peak_ratio"] <= thresholds["peak_ratio"])
            ].copy()
            if len(selected) < 100:
                thresholds = {
                    "std_log": _safe_quantile(region["std_log"].to_numpy(), flat_relaxed_quantile),
                    "range_log": _safe_quantile(region["range_log"].to_numpy(), flat_relaxed_quantile),
                    "peak_ratio": _safe_quantile(region["peak_ratio"].to_numpy(), flat_relaxed_quantile),
                }
                selected = region[
                    (region["std_log"] <= thresholds["std_log"])
                    & (region["range_log"] <= thresholds["range_log"])
                    & (region["peak_ratio"] <= thresholds["peak_ratio"])
                ].copy()
            d2_parts.append(selected)
        if d2_parts:
            d2 = pd.concat(d2_parts, ignore_index=True).drop_duplicates(["case_name", "node_index"])
            d2["pointset_type"] = "full_sensitive_flat"
            rows.append(_cap_per_case(d2, max_nodes_per_case_pointset, seed + 1))

    max_threshold = _safe_quantile(stats_df.loc[non_bc, "max_raw"].to_numpy(), hotspot_max_quantile)
    std_threshold = _safe_quantile(stats_df.loc[non_bc, "std_log"].to_numpy(), hotspot_dynamic_quantile)
    peak_ratio_threshold = _safe_quantile(stats_df.loc[non_bc, "peak_ratio"].to_numpy(), hotspot_dynamic_quantile)
    d3 = stats_df[
        non_bc
        & (
            (stats_df["frame_top_candidate"] == 1)
            | (
                (stats_df["max_raw"] >= max_threshold)
                & (stats_df["std_log"] >= std_threshold)
                & (stats_df["peak_ratio"] >= peak_ratio_threshold)
            )
        )
    ].copy()
    if not d3.empty:
        d3["pointset_type"] = "full_hotspot"
        rows.append(_cap_per_case(d3, max_nodes_per_case_pointset, seed + 2))

    if not rows:
        return pd.DataFrame()
    selected = pd.concat(rows, ignore_index=True)
    selected = selected.drop_duplicates(["case_name", "node_index", "pointset_type"])
    return selected


def _cap_per_case(df: pd.DataFrame, max_nodes_per_case: int, seed: int) -> pd.DataFrame:
    if max_nodes_per_case <= 0:
        return df
    parts = [
        _balanced_take(group, max_nodes_per_case, seed + idx)
        for idx, (_, group) in enumerate(df.groupby("case_name", sort=False))
    ]
    return pd.concat(parts, ignore_index=True) if parts else df


def _summarize_pointsets(selected: pd.DataFrame, frequency_counts: dict[str, int]) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (split, pointset_type, region_type), group in selected.groupby(["split", "pointset_type", "region_type"], dropna=False):
        freq_counts = [int(frequency_counts.get(str(case_name), 0)) for case_name in group["case_name"].unique()]
        rows.append(
            {
                "split": split,
                "pointset_type": pointset_type,
                "region_type": region_type,
                "case_count": int(group["case_name"].nunique()),
                "case_node_count": int(len(group[["case_name", "node_index"]].drop_duplicates())),
                "case_frequency_point_count": int(
                    sum(int(frequency_counts.get(str(row.case_name), 0)) for row in group.itertuples(index=False))
                ),
                "frequency_count_mean": float(np.mean(freq_counts)) if freq_counts else 0.0,
                "frequency_count_min": int(np.min(freq_counts)) if freq_counts else 0,
                "mean_log_p50": _safe_quantile(group["mean_log"].to_numpy(), 0.50),
                "std_log_p50": _safe_quantile(group["std_log"].to_numpy(), 0.50),
                "peak_ratio_p50": _safe_quantile(group["peak_ratio"].to_numpy(), 0.50),
                "max_raw_p90": _safe_quantile(group["max_raw"].to_numpy(), 0.90),
            }
        )
    return pd.DataFrame(rows)


def _write_data_description(output_dir: Path, selected: pd.DataFrame, summary: pd.DataFrame, config_path: str) -> None:
    lines = [
        "# 全零件三类点诊断实验数据说明",
        "",
        "本目录由 `node/build_fullpart_diagnostic_pointsets.py` 生成，用于训练 D1/D2/D3 specialist 诊断模型。",
        "",
        f"- 基础配置：`{config_path}`",
        "- 点集范围：全零件节点，不限于耳片区域。",
        "- 训练样本展开方式：选中的 `(case, node)` 会在训练时展开为该 case 的所有频点样本。",
        "",
        "## 点集定义",
        "",
        "- `full_flat_background`：远离 `nodes.csv` 标注应力集中区域，且真实频谱平坦的背景点。",
        "- `full_sensitive_flat`：位于 `nodes.csv` 标注应力集中区域，但真实频谱平坦的点。",
        "- `full_hotspot`：真实响应峰值、top5% 或节点谱动态明显的热点点。",
        "",
        "## 应力集中区域",
        "",
        "- `center_couple_region`: `center_couple_mask`",
        "- `plate_hole_region`: `plate_hole_wall_mask`",
        "- `ear_hole_region`: `ear_hole_wall_mask`",
        "- `ear_connection_region`: `ear_connection_fillet_mask` / `ear_connection_earside_mask` / `ear_connection_mask`",
        "",
        "## 关键文件",
        "",
        "- `pointsets/selected_points_all.csv`: 所有 split 和点集的选点明细。",
        "- `pointsets/selected_points_train.csv`: train split 选点。",
        "- `pointsets/selected_points_val.csv`: val split 选点。",
        "- `pointsets/selected_points_test.csv`: test split 选点。",
        "- `metrics/diagnostic_pointset_summary.csv`: 按 split、点集、区域统计的数据量和响应分布。",
        "- `plots/pointset_counts_by_split.png`: 三类点数量对比。",
        "- `plots/pointset_region_counts.png`: 三类点在各应力集中区域的分布。",
        "- `plots/pointset_response_distributions.png`: 三类点的响应统计分布。",
        "",
        "## 数据量总览",
        "",
    ]
    if not summary.empty:
        totals = (
            summary.groupby(["split", "pointset_type"], as_index=False)["case_node_count"]
            .sum()
            .sort_values(["split", "pointset_type"])
        )
        lines.extend(_dataframe_to_markdown_lines(totals))
    else:
        lines.append("当前未选出点集。")
    lines.append("")
    lines.append("## 训练注意")
    lines.append("")
    lines.append("训练配置需要设置 `dataset.node_scope: all_nodes`，并通过 `dataset.pointset.path` 指向对应 split 的 CSV。")
    (output_dir / "DATA说明.md").write_text("\n".join(lines), encoding="utf-8")


def _dataframe_to_markdown_lines(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max([len(columns[idx]), *[len(row[idx]) for row in rows]] or [len(columns[idx])])
        for idx in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [fmt_row(columns), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt_row(row) for row in rows)
    return lines


def _plot_outputs(output_dir: Path, selected: pd.DataFrame) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    if selected.empty:
        return

    counts = selected.groupby(["split", "pointset_type"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Selected case-node counts by split")
    ax.set_xlabel("split")
    ax.set_ylabel("case-node count")
    ax.legend(title="pointset")
    plt.tight_layout()
    plt.savefig(plots_dir / "pointset_counts_by_split.png", dpi=160)
    plt.close()

    region_counts = selected.groupby(["pointset_type", "region_type"]).size().unstack(fill_value=0)
    ax = region_counts.plot(kind="bar", stacked=True, figsize=(11, 5))
    ax.set_title("Selected case-node counts by region")
    ax.set_xlabel("pointset")
    ax.set_ylabel("case-node count")
    ax.legend(title="region", fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "pointset_region_counts.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column in zip(axes, ["mean_log", "std_log", "peak_ratio"], strict=True):
        for pointset_type, group in selected.groupby("pointset_type"):
            values = group[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=50, histtype="step", density=True, label=pointset_type)
        ax.set_title(column)
        ax.set_ylabel("density")
    axes[-1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "pointset_response_distributions.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    config = read_config(args.config)
    dataset_cfg = dict(config["dataset"])
    output_dir = ensure_dir(args.output_dir)
    pointsets_dir = ensure_dir(output_dir / "pointsets")
    metrics_dir = ensure_dir(output_dir / "metrics")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.diagnostic_pointsets", log_file="build_pointsets.log")

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = _split_case_names(config, max_cases_per_split=args.max_cases_per_split)
    selected_parts: list[pd.DataFrame] = []
    frequency_counts: dict[str, int] = {}

    logger.info("Building full-part diagnostic pointsets | output=%s", output_dir)
    for split, case_names in split_names.items():
        logger.info("Split %s | cases=%s", split, len(case_names))
        split_stats: list[pd.DataFrame] = []
        for idx, case_name in enumerate(case_names, start=1):
            spectrum = _load_case_spectrum(case_index[case_name], dataset_cfg, max_frames_per_case=args.max_frames_per_case)
            frequency_counts[case_name] = int(spectrum.frequencies.size)
            case_stats = _case_node_stats(spectrum, top_fraction=float(args.top_fraction))
            split_stats.append(case_stats)
            if idx == 1 or idx == len(case_names) or idx % 20 == 0:
                logger.info(
                    "Pointset stats progress | split=%s | case=%s/%s | last=%s | nodes=%s | frames=%s",
                    split,
                    idx,
                    len(case_names),
                    case_name,
                    len(case_stats),
                    spectrum.frequencies.size,
                )
        if not split_stats:
            continue
        split_stats_df = pd.concat(split_stats, ignore_index=True)
        split_stats_df["split"] = split
        selected = _select_pointsets_for_split(
            split_stats_df,
            flat_quantile=float(args.flat_quantile),
            flat_relaxed_quantile=float(args.flat_relaxed_quantile),
            background_distance_quantile=float(args.background_distance_quantile),
            near_region_quantile=float(args.near_region_quantile),
            hotspot_max_quantile=float(args.hotspot_max_quantile),
            hotspot_dynamic_quantile=float(args.hotspot_dynamic_quantile),
            max_nodes_per_case_pointset=int(args.max_nodes_per_case_pointset),
            seed=int(args.seed),
        )
        if not selected.empty:
            selected["split"] = split
            selected_parts.append(selected)
            logger.info("Selected split %s | rows=%s", split, len(selected))

    selected_all = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if not selected_all.empty:
        selected_all = selected_all.sort_values(["split", "pointset_type", "case_name", "node_index"])
        output_columns = [
            "split",
            "pointset_type",
            "case_name",
            "node_index",
            "region_type",
            "stress_region_mask",
            "bc_mask",
            "center_node_mask",
            "valid_frequency_count",
            "mean_log",
            "std_log",
            "range_log",
            "mean_raw",
            "max_raw",
            "peak_ratio",
            "peak_frequency",
            "nearest_stress_distance",
            "frame_top_candidate",
        ]
        selected_all[output_columns].to_csv(pointsets_dir / "selected_points_all.csv", index=False)
        for split, group in selected_all.groupby("split", sort=False):
            group[output_columns].to_csv(pointsets_dir / f"selected_points_{split}.csv", index=False)
        for pointset_type, group in selected_all.groupby("pointset_type", sort=False):
            group[output_columns].to_csv(pointsets_dir / f"selected_points_{pointset_type}.csv", index=False)
    if bool(args.write_all_node_stats):
        logger.warning(
            "write-all-node-stats is enabled; output can be very large for the full 1000-case dataset."
        )
        # Rebuild a compact selected-only stats file by default. Full per-node stats
        # should only be used in small debug runs.
        selected_all.to_csv(metrics_dir / "selected_case_node_stats.csv", index=False)

    summary = _summarize_pointsets(selected_all, frequency_counts=frequency_counts)
    summary.to_csv(metrics_dir / "diagnostic_pointset_summary.csv", index=False)
    _plot_outputs(output_dir, selected_all)
    _write_data_description(output_dir, selected_all, summary, config_path=args.config)
    write_json(
        metrics_dir / "build_config.json",
        {
            "config": args.config,
            "output_dir": str(output_dir),
            "max_cases_per_split": args.max_cases_per_split,
            "max_frames_per_case": args.max_frames_per_case,
            "flat_quantile": args.flat_quantile,
            "flat_relaxed_quantile": args.flat_relaxed_quantile,
            "background_distance_quantile": args.background_distance_quantile,
            "hotspot_max_quantile": args.hotspot_max_quantile,
            "hotspot_dynamic_quantile": args.hotspot_dynamic_quantile,
            "top_fraction": args.top_fraction,
            "max_nodes_per_case_pointset": args.max_nodes_per_case_pointset,
            "write_all_node_stats": bool(args.write_all_node_stats),
            "seed": args.seed,
        },
    )
    logger.info("Saved pointsets: %s", pointsets_dir)
    logger.info("Saved summary: %s", metrics_dir / "diagnostic_pointset_summary.csv")


if __name__ == "__main__":
    main()

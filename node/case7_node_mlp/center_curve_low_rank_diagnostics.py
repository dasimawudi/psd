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
    _frequency_from_path,
    _load_aligned_target_column,
    _load_case_static,
    build_node_selection_mask,
    discover_case_index,
    expand_case_sample_paths,
    resolve_case_splits,
)
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, write_json
from case7_node_mlp.trainer import _apply_target_floor, _resolve_threshold_from_config


SUMMARY_FIELDS = [
    "split",
    "case_name",
    "frequencies",
    "nodes",
    "hotspot_type",
    "top1_unique_nodes",
    "top5_union_nodes",
    "top1_radius_mean",
    "rank1_explained",
    "rank2_explained",
    "rank3_explained",
    "rank5_explained",
    "rank10_explained",
    "mean_pairwise_corr",
    "median_pairwise_corr",
    "min_pairwise_corr",
    "top1_mean_abs_pc1_weight",
    "top5_mean_abs_pc1_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose low-rank frequency-curve structure in the disk center region.")
    parser.add_argument("--config", type=str, default="node/configs/node_mlp_v6_disk_center_hotspot_features.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default="node/outputs/center_curve_low_rank_diagnostics")
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--target-floor", type=str, default=None)
    return parser.parse_args()


def _target_floor(config: dict[str, Any], schema: dict[str, Any], override: str | None) -> float:
    if override is not None:
        local = dict(config.get("target", {}))
        local["zero_below"] = override
        return float(_resolve_threshold_from_config(local, schema, "zero_below"))
    return float(_resolve_threshold_from_config(dict(config.get("target", {})), schema, "zero_below"))


def _target_floor_needs_stats(config: dict[str, Any], override: str | None) -> bool:
    raw_value = override if override is not None else dict(config.get("target", {})).get("zero_below", 0.0)
    if raw_value is None:
        return False
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        return value.startswith("p95*") or value.startswith("p99*")
    return False


def _load_target_stats(sample_paths: list[Path], dataset_cfg: dict[str, Any], max_values: int = 500_000) -> dict[str, float]:
    values: list[np.ndarray] = []
    per_file_cap = max(1, int(max_values) // max(len(sample_paths), 1))
    for sample_path in sample_paths:
        case_dir = sample_path.parent.parent
        nodes_df, _payload, _earpiece_mask = _load_case_static(
            case_dir,
            region_cfg=dataset_cfg.get("earpiece_region"),
        )
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
        selected = target_values[selected_mask]
        selected = selected[np.isfinite(selected) & (selected > 0.0)]
        if selected.size > per_file_cap:
            rng = np.random.default_rng(abs(hash(str(sample_path))) % (2**32))
            selected = selected[rng.choice(selected.size, size=per_file_cap, replace=False)]
        if selected.size:
            values.append(selected.astype(np.float64, copy=False))
    if not values:
        return {}
    positive = np.concatenate(values)
    return {
        "target_positive_p95": float(np.quantile(positive, 0.95)),
        "target_positive_p99": float(np.quantile(positive, 0.99)),
        "target_positive_p999": float(np.quantile(positive, 0.999)),
    }


def _node_radius(nodes_df: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    x = nodes_df["x"].to_numpy(dtype=np.float64, copy=False)[positions]
    y = nodes_df["y"].to_numpy(dtype=np.float64, copy=False)[positions]
    return np.sqrt(x * x + y * y)


def _select_cases(case_names: list[str], requested: list[str] | None, num_cases: int, seed: int) -> list[str]:
    if requested:
        requested_set = set(requested)
        return [name for name in case_names if name in requested_set]
    rng = np.random.default_rng(int(seed))
    names = list(case_names)
    if num_cases > 0 and len(names) > num_cases:
        return sorted(rng.choice(names, size=num_cases, replace=False).tolist())
    return names


def _build_case_matrix(
    *,
    case_dir: Path,
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    target_floor: float,
    max_nodes: int,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes_df, _payload, _earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    selected_mask = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg)
    node_positions = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    if node_positions.size == 0:
        raise ValueError(f"No selected nodes for {case_dir.name}")

    frame_rows: list[np.ndarray] = []
    frequencies: list[float] = []
    top_nodes: set[int] = set()
    top5_union: set[int] = set()
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
        if not np.isfinite(selected_values).any():
            continue
        frame_rows.append(np.nan_to_num(selected_values.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
        frequencies.append(float(_frequency_from_path(sample_path)))
        order = np.argsort(selected_values)[::-1]
        if order.size:
            top_nodes.add(int(node_positions[int(order[0])]))
            top_count = max(1, int(math.ceil(order.size * min(max(top_fraction, 0.0), 1.0))))
            for local_index in order[:top_count]:
                top5_union.add(int(node_positions[int(local_index)]))

    if not frame_rows:
        raise ValueError(f"No valid frames for {case_dir.name}")
    raw_matrix = np.stack(frame_rows, axis=1)
    node_peak = raw_matrix.max(axis=1)
    if max_nodes > 0 and raw_matrix.shape[0] > max_nodes:
        keep = np.argsort(node_peak)[-max_nodes:]
        keep = np.sort(keep)
    else:
        keep = np.arange(raw_matrix.shape[0])
    matrix = np.log1p(raw_matrix[keep])
    selected_positions = node_positions[keep]
    radii = _node_radius(nodes_df, selected_positions)
    top1_positions = np.array(sorted(top_nodes), dtype=np.int64)
    top5_positions = np.array(sorted(top5_union), dtype=np.int64)
    return matrix, np.asarray(frequencies, dtype=np.float64), selected_positions, radii, top1_positions, top5_positions


def _svd_summary(matrix: np.ndarray, top1_positions: np.ndarray, top5_positions: np.ndarray, selected_positions: np.ndarray) -> dict[str, float]:
    row_centered = matrix - matrix.mean(axis=1, keepdims=True)
    centered = row_centered - row_centered.mean(axis=0, keepdims=True)
    if min(centered.shape) <= 1 or float(np.linalg.norm(centered)) <= 1e-12:
        return {key: 0.0 for key in ("rank1_explained", "rank2_explained", "rank3_explained", "rank5_explained", "rank10_explained")} | {
            "mean_pairwise_corr": float("nan"),
            "median_pairwise_corr": float("nan"),
            "min_pairwise_corr": float("nan"),
            "top1_mean_abs_pc1_weight": float("nan"),
            "top5_mean_abs_pc1_weight": float("nan"),
        }

    u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular_values * singular_values
    total = float(energy.sum())

    def explained(rank: int) -> float:
        return float(energy[: min(rank, energy.size)].sum() / total) if total > 0.0 else 0.0

    normalized = row_centered
    denom = np.linalg.norm(normalized, axis=1, keepdims=True)
    valid = denom.reshape(-1) > 1e-12
    corr_values = np.array([], dtype=np.float64)
    if int(valid.sum()) >= 2:
        unit = normalized[valid] / denom[valid]
        corr = unit @ unit.T
        corr_values = corr[np.triu_indices(corr.shape[0], k=1)]
    position_to_row = {int(position): row for row, position in enumerate(selected_positions.tolist())}
    top1_rows = [position_to_row[int(position)] for position in top1_positions if int(position) in position_to_row]
    top5_rows = [position_to_row[int(position)] for position in top5_positions if int(position) in position_to_row]
    pc1_weights = np.abs(u[:, 0]) if u.size else np.array([], dtype=np.float64)
    return {
        "rank1_explained": explained(1),
        "rank2_explained": explained(2),
        "rank3_explained": explained(3),
        "rank5_explained": explained(5),
        "rank10_explained": explained(10),
        "mean_pairwise_corr": float(np.mean(corr_values)) if corr_values.size else float("nan"),
        "median_pairwise_corr": float(np.median(corr_values)) if corr_values.size else float("nan"),
        "min_pairwise_corr": float(np.min(corr_values)) if corr_values.size else float("nan"),
        "top1_mean_abs_pc1_weight": float(np.mean(pc1_weights[top1_rows])) if top1_rows else float("nan"),
        "top5_mean_abs_pc1_weight": float(np.mean(pc1_weights[top5_rows])) if top5_rows else float("nan"),
    }


def _plot_singular_values(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    ranks = [1, 2, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary_rows:
        values = [float(row.get(f"rank{rank}_explained", 0.0)) for rank in ranks]
        ax.plot(ranks, values, alpha=0.20, linewidth=1.0, color="#4c78a8")
    medians = [float(np.nanmedian([float(row.get(f"rank{rank}_explained", np.nan)) for row in summary_rows])) for rank in ranks]
    ax.plot(ranks, medians, marker="o", linewidth=2.2, color="#d62728", label="median")
    ax.set_xlabel("rank")
    ax.set_ylabel("explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Disk-center target curve low-rank structure")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_diagnostics(
    *,
    config_path: str | Path,
    split: str,
    output_dir: str | Path,
    num_cases: int,
    case_names: list[str] | None,
    seed: int,
    max_frames_per_case: int | None,
    max_nodes: int,
    top_fraction: float,
    target_floor_override: str | None,
) -> dict[str, Any]:
    config = read_config(config_path)
    output_path = ensure_dir(output_dir)
    logger = make_logger(output_path, logger_name="case7_node_mlp.center_curve_low_rank_diagnostics", log_file="low_rank_diagnostics.log")
    dataset_cfg = dict(config.get("dataset", {}))
    if max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(max_frames_per_case)
    case_index = discover_case_index(dataset_cfg["root"])
    splits = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    selected_case_names = _select_cases(splits.get(split, []), case_names, num_cases=num_cases, seed=seed)
    selected_case_dirs = [case_index[name] for name in selected_case_names]
    all_sample_paths = expand_case_sample_paths(selected_case_dirs, dataset_cfg)
    target_stats = _load_target_stats(all_sample_paths, dataset_cfg) if _target_floor_needs_stats(config, target_floor_override) else {}
    floor = _target_floor(config, target_stats, target_floor_override)
    rows: list[dict[str, Any]] = []
    for offset, case_dir in enumerate(selected_case_dirs, start=1):
        sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
        logger.info("Case %s/%s | %s | frames=%s", offset, len(selected_case_dirs), case_dir.name, len(sample_paths))
        matrix, frequencies, positions, radii, top1_positions, top5_positions = _build_case_matrix(
            case_dir=case_dir,
            sample_paths=sample_paths,
            dataset_cfg=dataset_cfg,
            target_floor=floor,
            max_nodes=max_nodes,
            top_fraction=top_fraction,
        )
        svd = _svd_summary(matrix, top1_positions=top1_positions, top5_positions=top5_positions, selected_positions=positions)
        top1_radii = radii[np.isin(positions, top1_positions)]
        hotspot_type = "outer_ring" if top1_radii.size and float(np.nanmedian(top1_radii)) >= 12.0 else "center_core"
        row = {
            "split": split,
            "case_name": case_dir.name,
            "frequencies": int(frequencies.size),
            "nodes": int(matrix.shape[0]),
            "hotspot_type": hotspot_type,
            "top1_unique_nodes": int(top1_positions.size),
            "top5_union_nodes": int(top5_positions.size),
            "top1_radius_mean": float(np.nanmean(top1_radii)) if top1_radii.size else float("nan"),
            **svd,
        }
        rows.append(row)

    csv_path = output_path / f"{split}_low_rank_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _plot_singular_values(output_path / f"{split}_rank_explained.png", rows)
    aggregate: dict[str, Any] = {
        "config": str(config_path),
        "split": split,
        "cases": len(rows),
        "target_floor": floor,
        "summary_csv": str(csv_path),
    }
    for field in ["rank1_explained", "rank2_explained", "rank3_explained", "rank5_explained", "mean_pairwise_corr", "median_pairwise_corr"]:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        aggregate[f"{field}_median"] = float(np.median(values)) if values.size else float("nan")
        aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
    write_json(output_path / f"{split}_low_rank_summary.json", aggregate)
    logger.info("Saved low-rank diagnostics: %s", output_path)
    return aggregate


def main() -> None:
    args = parse_args()
    run_diagnostics(
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
        num_cases=int(args.num_cases),
        case_names=args.case_name,
        seed=int(args.seed),
        max_frames_per_case=args.max_frames_per_case,
        max_nodes=int(args.max_nodes),
        top_fraction=float(args.top_fraction),
        target_floor_override=args.target_floor,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from case7_node_mlp.trainer import _resolve_threshold_from_config


DEFAULT_ROOT = ".cache/mises_psd_next1000_full_export_step0p5_balanced_v1"

SUMMARY_FIELDNAMES = [
    "curve_id",
    "split",
    "group",
    "case_name",
    "node_position",
    "node_index",
    "node_label",
    "points",
    "case_curve_count",
    "peak_rank",
    "peak_rank_fraction",
    "peak_target",
    "peak_frequency_hz",
    "mean_target",
    "std_target",
]

POINT_FIELDNAMES = [
    "curve_id",
    "split",
    "group",
    "case_name",
    "node_position",
    "node_index",
    "node_label",
    "frequency_hz",
    "target_raw",
    "target_log",
    "is_curve_peak_frequency",
    "peak_target",
    "peak_frequency_hz",
    "peak_rank_fraction",
]

CASE_COUNT_FIELDNAMES = [
    "split",
    "case_name",
    "frames",
    "selected_nodes",
    "eligible_curves",
    "hotspot_candidates",
    "non_hotspot_candidates",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample full-frequency node curves from raw per-frequency MISES targets. "
            "Each curve is one case/node traced across all frequency frames."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint. Its embedded config/splits and target floor are reused unless --config/--root override them.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config with dataset settings.")
    parser.add_argument("--root", type=str, default=None, help=f"Dataset root. Default: {DEFAULT_ROOT}")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of curves to sample per group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--hotspot-fraction",
        type=float,
        default=0.05,
        help="Curve-level hotspot threshold: top fraction by per-node peak response inside each case.",
    )
    parser.add_argument(
        "--non-hotspot-min-rank-fraction",
        type=float,
        default=0.50,
        help="Curve-level non-hotspot threshold: nodes with peak-rank fraction at or below this response tier.",
    )
    parser.add_argument(
        "--min-frequency-coverage",
        type=float,
        default=0.80,
        help="Minimum fraction of case frames a node must have valid targets for.",
    )
    parser.add_argument("--min-peak", type=float, default=0.0, help="Drop curves with peak response below this value.")
    parser.add_argument("--target-floor", type=float, default=None, help="Override target.zero_below used before ranking/writing.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional quick-run case limit.")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional quick-run frame limit per case.")
    parser.add_argument(
        "--plot-count-per-group",
        type=int,
        default=0,
        help="Write PNG plots for the first N sampled curves in each group. Default writes CSV only.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")

    if args.config:
        config = read_config(args.config)
    elif checkpoint:
        config = dict(checkpoint["config"])
    else:
        config = {
            "dataset": {
                "root": args.root or DEFAULT_ROOT,
                "sample_mode": "per_frequency",
                "include_zero_frequency": False,
                "min_frequency_hz": 20.0,
                "max_frequency_hz": 2000.0,
                "split_mode": "ratio",
                "split_seed": 42,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "earpiece_region": {"enabled": True, "type": "earpiece", "width_scale": 1.5},
            },
            "target": {},
        }

    dataset_cfg = dict(config.get("dataset", {}))
    if args.root:
        dataset_cfg["root"] = args.root
    if "root" not in dataset_cfg:
        dataset_cfg["root"] = DEFAULT_ROOT
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)
    config["dataset"] = dataset_cfg
    return config, checkpoint


def _resolve_target_floor(
    args: argparse.Namespace,
    target_cfg: dict[str, Any],
    checkpoint: dict[str, Any],
) -> float:
    if args.target_floor is not None:
        return float(args.target_floor)
    feature_schema = dict(checkpoint.get("feature_schema", {}))
    if feature_schema:
        return float(_resolve_threshold_from_config(target_cfg, feature_schema, "zero_below"))
    raw_value = target_cfg.get("zero_below", 0.0)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"", "none", "off", "false"}:
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    return float(raw_value or 0.0)


def _apply_target_floor_np(values: np.ndarray, target_floor: float) -> np.ndarray:
    output = np.clip(values.astype(np.float64, copy=False), 0.0, None)
    if target_floor > 0.0:
        output = np.where(output < float(target_floor), 0.0, output)
    return output


def _load_aligned_targets(case_dir: Path, nodes_df: pd.DataFrame, sample_path: Path, target_floor: float) -> np.ndarray:
    target_df = pd.read_csv(
        sample_path,
        usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN},
    )
    values = _load_aligned_target_column(
        target_df,
        target_column=PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_df,
        target_path=sample_path,
    )
    if values.shape[0] != len(nodes_df):
        raise ValueError(f"{sample_path} target count {values.shape[0]} does not match nodes count {len(nodes_df)}")
    return _apply_target_floor_np(values, target_floor=target_floor)


def _node_identity(nodes_df: pd.DataFrame, node_position: int) -> tuple[int, int]:
    row = nodes_df.iloc[int(node_position)]
    node_index = int(row["node_index"]) if "node_index" in nodes_df.columns else int(node_position)
    if "node_label" in nodes_df.columns:
        node_label = int(row["node_label"])
    else:
        node_label = node_index
    return node_index, node_label


def _reservoir_update(
    reservoir: list[dict[str, Any]],
    candidate: dict[str, Any],
    seen_count: int,
    sample_size: int,
    rng: np.random.Generator,
) -> None:
    if sample_size <= 0:
        return
    if len(reservoir) < sample_size:
        reservoir.append(candidate)
        return
    replacement_index = int(rng.integers(0, seen_count))
    if replacement_index < sample_size:
        reservoir[replacement_index] = candidate


def _scan_case_candidates(
    *,
    split: str,
    case_dir: Path,
    dataset_cfg: dict[str, Any],
    sample_paths: list[Path],
    target_floor: float,
    hotspot_fraction: float,
    non_hotspot_min_rank_fraction: float,
    min_frequency_coverage: float,
    min_peak: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes_df, _payload, _earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    selection_mask = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg)
    selected_positions = np.flatnonzero(selection_mask).astype(np.int64, copy=False)
    if selected_positions.size == 0:
        case_counts = {
            "split": split,
            "case_name": case_dir.name,
            "frames": len(sample_paths),
            "selected_nodes": 0,
            "eligible_curves": 0,
            "hotspot_candidates": 0,
            "non_hotspot_candidates": 0,
        }
        return [], [], case_counts

    selected_count = int(selected_positions.size)
    counts = np.zeros(selected_count, dtype=np.int32)
    sums = np.zeros(selected_count, dtype=np.float64)
    sumsq = np.zeros(selected_count, dtype=np.float64)
    peaks = np.full(selected_count, -np.inf, dtype=np.float64)
    peak_frequencies = np.full(selected_count, np.nan, dtype=np.float64)

    for sample_path in sample_paths:
        frequency_hz = _frequency_from_path(sample_path)
        values = _load_aligned_targets(case_dir, nodes_df, sample_path, target_floor)[selected_positions]
        valid = np.isfinite(values) & (values >= 0.0)
        if not bool(valid.any()):
            continue
        valid_values = values[valid]
        counts[valid] += 1
        sums[valid] += valid_values
        sumsq[valid] += np.square(valid_values)
        better_peak = valid & (values > peaks)
        peaks[better_peak] = values[better_peak]
        peak_frequencies[better_peak] = float(frequency_hz)

    min_count = max(1, int(math.ceil(float(min_frequency_coverage) * max(len(sample_paths), 1))))
    eligible_mask = (counts >= min_count) & np.isfinite(peaks) & (peaks >= float(min_peak))
    eligible_indices = np.flatnonzero(eligible_mask)
    if eligible_indices.size == 0:
        case_counts = {
            "split": split,
            "case_name": case_dir.name,
            "frames": len(sample_paths),
            "selected_nodes": selected_count,
            "eligible_curves": 0,
            "hotspot_candidates": 0,
            "non_hotspot_candidates": 0,
        }
        return [], [], case_counts

    order = eligible_indices[np.argsort(-peaks[eligible_indices], kind="mergesort")]
    case_curve_count = int(order.size)
    hotspot_rows: list[dict[str, Any]] = []
    non_hotspot_rows: list[dict[str, Any]] = []

    for zero_based_rank, selected_index in enumerate(order):
        peak_rank = int(zero_based_rank + 1)
        rank_fraction = float(peak_rank / max(case_curve_count, 1))
        if rank_fraction > float(hotspot_fraction) and rank_fraction < float(non_hotspot_min_rank_fraction):
            continue
        node_position = int(selected_positions[int(selected_index)])
        node_index, node_label = _node_identity(nodes_df, node_position)
        count = int(counts[selected_index])
        mean_target = float(sums[selected_index] / max(count, 1))
        variance = max(float(sumsq[selected_index] / max(count, 1) - mean_target * mean_target), 0.0)
        row = {
            "split": split,
            "case_name": case_dir.name,
            "node_position": node_position,
            "node_index": node_index,
            "node_label": node_label,
            "points": count,
            "case_curve_count": case_curve_count,
            "peak_rank": peak_rank,
            "peak_rank_fraction": rank_fraction,
            "peak_target": float(peaks[selected_index]),
            "peak_frequency_hz": float(peak_frequencies[selected_index]),
            "mean_target": mean_target,
            "std_target": float(math.sqrt(variance)),
        }
        if rank_fraction <= float(hotspot_fraction):
            hotspot_rows.append({**row, "group": "hotspot"})
        elif rank_fraction >= float(non_hotspot_min_rank_fraction):
            non_hotspot_rows.append({**row, "group": "non_hotspot"})

    case_counts = {
        "split": split,
        "case_name": case_dir.name,
        "frames": len(sample_paths),
        "selected_nodes": selected_count,
        "eligible_curves": case_curve_count,
        "hotspot_candidates": len(hotspot_rows),
        "non_hotspot_candidates": len(non_hotspot_rows),
    }
    return hotspot_rows, non_hotspot_rows, case_counts


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _assign_curve_ids(hotspot_rows: list[dict[str, Any]], non_hotspot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group_name, rows in (("hotspot", hotspot_rows), ("non_hotspot", non_hotspot_rows)):
        ordered = sorted(rows, key=lambda row: (str(row["case_name"]), int(row["node_position"])))
        for offset, row in enumerate(ordered, start=1):
            selected.append({**row, "curve_id": f"{group_name}_{offset:04d}"})
    return selected


def _write_curve_points(
    *,
    path: Path,
    selected_rows: list[dict[str, Any]],
    case_index: dict[str, Path],
    dataset_cfg: dict[str, Any],
    target_floor: float,
    plot_count_per_group: int,
    plot_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    curves_for_plot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plot_ids: set[str] = set()
    if plot_count_per_group > 0:
        for group_name in ("hotspot", "non_hotspot"):
            group_rows = [row for row in selected_rows if row["group"] == group_name]
            plot_ids.update(str(row["curve_id"]) for row in group_rows[: int(plot_count_per_group)])

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_case[str(row["case_name"])].append(row)

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=POINT_FIELDNAMES)
        writer.writeheader()
        for case_name in sorted(by_case):
            case_dir = case_index[case_name]
            nodes_df, _payload, _earpiece_mask = _load_case_static(
                case_dir,
                region_cfg=dataset_cfg.get("earpiece_region"),
            )
            sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
            case_rows = sorted(by_case[case_name], key=lambda row: str(row["curve_id"]))
            positions = np.array([int(row["node_position"]) for row in case_rows], dtype=np.int64)
            for sample_path in sample_paths:
                frequency_hz = _frequency_from_path(sample_path)
                values = _load_aligned_targets(case_dir, nodes_df, sample_path, target_floor)
                selected_values = values[positions]
                for curve_row, target_raw in zip(case_rows, selected_values):
                    safe_target = max(float(target_raw), 0.0)
                    point_row = {
                        "curve_id": curve_row["curve_id"],
                        "split": curve_row["split"],
                        "group": curve_row["group"],
                        "case_name": curve_row["case_name"],
                        "node_position": curve_row["node_position"],
                        "node_index": curve_row["node_index"],
                        "node_label": curve_row["node_label"],
                        "frequency_hz": float(frequency_hz),
                        "target_raw": safe_target,
                        "target_log": math.log1p(safe_target),
                        "is_curve_peak_frequency": float(
                            abs(float(frequency_hz) - float(curve_row["peak_frequency_hz"])) <= 1e-6
                        ),
                        "peak_target": curve_row["peak_target"],
                        "peak_frequency_hz": curve_row["peak_frequency_hz"],
                        "peak_rank_fraction": curve_row["peak_rank_fraction"],
                    }
                    writer.writerow(point_row)
                    if str(curve_row["curve_id"]) in plot_ids:
                        curves_for_plot[str(curve_row["curve_id"])].append(point_row)

    if curves_for_plot:
        _write_curve_plots(curves_for_plot, plot_dir)
    return curves_for_plot


def _write_curve_plots(curves_for_plot: dict[str, list[dict[str, Any]]], plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(plot_dir)
    for curve_id, rows in curves_for_plot.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("frequency_hz")
        first = rows[0]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df["frequency_hz"], df["target_raw"].clip(lower=1e-12), linewidth=1.8)
        ax.scatter(
            [float(first["peak_frequency_hz"])],
            [max(float(first["peak_target"]), 1e-12)],
            s=28,
            color="tab:red",
            label="curve peak",
        )
        ax.set_yscale("log")
        ax.set_xlabel("frequency Hz")
        ax.set_ylabel(PER_FREQUENCY_TARGET_COLUMN)
        ax.set_title(
            f"{curve_id} | {first['case_name']} | node={first['node_index']} | "
            f"{first['group']} | peak rank={float(first['peak_rank_fraction']) * 100.0:.2f}%"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{curve_id}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if int(args.sample_size) <= 0:
        raise ValueError("--sample-size must be positive.")
    if not (0.0 < float(args.hotspot_fraction) <= 1.0):
        raise ValueError("--hotspot-fraction must be in (0, 1].")
    if not (0.0 < float(args.non_hotspot_min_rank_fraction) <= 1.0):
        raise ValueError("--non-hotspot-min-rank-fraction must be in (0, 1].")
    if float(args.non_hotspot_min_rank_fraction) <= float(args.hotspot_fraction):
        raise ValueError("--non-hotspot-min-rank-fraction must be larger than --hotspot-fraction.")

    config, checkpoint = _load_config(args)
    dataset_cfg = dict(config["dataset"])
    target_cfg = dict(config.get("target", {}))
    target_floor = _resolve_target_floor(args, target_cfg, checkpoint)
    output_dir = ensure_dir(args.output_dir or Path("node/outputs") / f"random_frequency_curves_{args.split}_{args.sample_size}each")
    logger = make_logger(output_dir, logger_name="sample_random_frequency_curves", log_file="sample_random_frequency_curves.log")

    case_index = discover_case_index(dataset_cfg["root"])
    if args.split == "all":
        split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
        case_names = sorted({name for names in split_names.values() for name in names})
    else:
        split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
        case_names = list(split_names[args.split])
    if args.max_cases is not None:
        case_names = case_names[: int(args.max_cases)]
    if not case_names:
        raise RuntimeError(f"No cases found for split={args.split}.")

    logger.info(
        "Sampling curves | split=%s | cases=%s | sample_size=%s/group | hotspot_fraction=%.4f | non_hotspot_min_rank_fraction=%.4f | target_floor=%.6g",
        args.split,
        len(case_names),
        int(args.sample_size),
        float(args.hotspot_fraction),
        float(args.non_hotspot_min_rank_fraction),
        target_floor,
    )

    rng = np.random.default_rng(int(args.seed))
    hotspot_reservoir: list[dict[str, Any]] = []
    non_hotspot_reservoir: list[dict[str, Any]] = []
    hotspot_seen = 0
    non_hotspot_seen = 0
    case_count_rows: list[dict[str, Any]] = []

    for case_offset, case_name in enumerate(case_names, start=1):
        case_dir = case_index[case_name]
        sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
        hotspot_rows, non_hotspot_rows, case_counts = _scan_case_candidates(
            split=args.split,
            case_dir=case_dir,
            dataset_cfg=dataset_cfg,
            sample_paths=sample_paths,
            target_floor=target_floor,
            hotspot_fraction=float(args.hotspot_fraction),
            non_hotspot_min_rank_fraction=float(args.non_hotspot_min_rank_fraction),
            min_frequency_coverage=float(args.min_frequency_coverage),
            min_peak=float(args.min_peak),
        )
        case_count_rows.append(case_counts)
        for row in hotspot_rows:
            hotspot_seen += 1
            _reservoir_update(hotspot_reservoir, row, hotspot_seen, int(args.sample_size), rng)
        for row in non_hotspot_rows:
            non_hotspot_seen += 1
            _reservoir_update(non_hotspot_reservoir, row, non_hotspot_seen, int(args.sample_size), rng)
        logger.info(
            "Case %s/%s | %s | frames=%s | eligible=%s | hotspot_candidates=%s | non_hotspot_candidates=%s",
            case_offset,
            len(case_names),
            case_name,
            case_counts["frames"],
            case_counts["eligible_curves"],
            case_counts["hotspot_candidates"],
            case_counts["non_hotspot_candidates"],
        )

    selected_rows = _assign_curve_ids(hotspot_reservoir, non_hotspot_reservoir)
    summary_path = output_dir / "selected_curve_summary.csv"
    points_path = output_dir / "selected_curve_points.csv"
    case_counts_path = output_dir / "candidate_counts_by_case.csv"
    _write_rows(summary_path, selected_rows, SUMMARY_FIELDNAMES)
    _write_rows(case_counts_path, case_count_rows, CASE_COUNT_FIELDNAMES)
    plot_dir = output_dir / "plots"
    _write_curve_points(
        path=points_path,
        selected_rows=selected_rows,
        case_index=case_index,
        dataset_cfg=dataset_cfg,
        target_floor=target_floor,
        plot_count_per_group=int(args.plot_count_per_group),
        plot_dir=plot_dir,
    )

    selected_hotspot = sum(1 for row in selected_rows if row["group"] == "hotspot")
    selected_non_hotspot = sum(1 for row in selected_rows if row["group"] == "non_hotspot")
    run_summary = {
        "split": args.split,
        "seed": int(args.seed),
        "dataset_root": str(dataset_cfg["root"]),
        "cases_scanned": len(case_names),
        "sample_size_per_group": int(args.sample_size),
        "hotspot_fraction": float(args.hotspot_fraction),
        "non_hotspot_min_rank_fraction": float(args.non_hotspot_min_rank_fraction),
        "min_frequency_coverage": float(args.min_frequency_coverage),
        "min_peak": float(args.min_peak),
        "target_floor": target_floor,
        "hotspot_candidates_seen": hotspot_seen,
        "non_hotspot_candidates_seen": non_hotspot_seen,
        "selected_hotspot_curves": selected_hotspot,
        "selected_non_hotspot_curves": selected_non_hotspot,
        "selected_curve_summary_csv": str(summary_path),
        "selected_curve_points_csv": str(points_path),
        "candidate_counts_by_case_csv": str(case_counts_path),
        "plot_dir": str(plot_dir) if int(args.plot_count_per_group) > 0 else None,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    logger.info("Wrote selected curve summary: %s", summary_path)
    logger.info("Wrote selected curve points: %s", points_path)
    logger.info("Wrote case candidate counts: %s", case_counts_path)
    logger.info("Selected curves | hotspot=%s/%s | non_hotspot=%s/%s", selected_hotspot, hotspot_seen, selected_non_hotspot, non_hotspot_seen)


if __name__ == "__main__":
    main()

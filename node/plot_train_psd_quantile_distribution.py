from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NODE_ROOT = Path(__file__).resolve().parent
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

from case7_node_mlp.data import (  # noqa: E402
    PER_FREQUENCY_TARGET_COLUMN,
    _build_boundary_mask,
    _build_disk_center_region_mask,
    _build_earpiece_region_mask,
    _frequency_from_path,
    _load_aligned_target_column,
    _load_global_payload,
    discover_case_index,
    expand_case_sample_paths,
    resolve_case_splits,
)
from case7_node_mlp.runtime import read_config  # noqa: E402


REGIONS = {
    "disk_center": {"label": "Disk Center", "color": "#C23B32"},
    "earpiece": {"label": "Earpiece", "color": "#2F6F9F"},
    "fullpart": {"label": "Full Part", "color": "#5E7F6E"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 1-percentile training-data PSD distributions for disk center, "
            "earpiece, and full-part regions."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="node/configs/node_mlp_v6_disk_center_low_rank_curves_fast_eval_continue70_4m.yaml",
        help="Training config used to resolve dataset root, split, and frequency filters.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="node/outputs/node_mlp_v6_disk_center_low_rank_curves_fast_eval_continue70_4m/train_psd_quantile_distribution_p1",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test limit after split selection.")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional override for frequency frames per case.")
    parser.add_argument(
        "--sample-values-per-file",
        type=int,
        default=128,
        help="Deterministic per-region samples per frame for estimating 1% quantile edges.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--exclude-bc-nodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override boundary-node exclusion. Defaults to dataset.exclude_bc_nodes.",
    )
    parser.add_argument(
        "--combined-plot-name",
        type=str,
        default="train_psd_p1_distribution_all_regions.png",
        help="Filename for the combined three-region plot.",
    )
    parser.add_argument(
        "--histogram-only",
        action="store_true",
        help="Generate PSD value histograms with fixed log-spaced value bins instead of percentile-bin plots.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=100,
        help="Number of positive PSD value bins for --histogram-only.",
    )
    parser.add_argument(
        "--histogram-combined-plot-name",
        type=str,
        default="train_psd_value_histogram_100bins_all_regions.png",
        help="Filename for the combined fixed-value-bin histogram plot.",
    )
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    abs_value = abs(float(value))
    if abs_value == 0.0:
        return "0"
    if abs_value < 1e-2 or abs_value >= 1e5:
        return f"{value:.2e}"
    if abs_value < 10.0:
        return f"{value:.2f}"
    if abs_value < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


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


def _load_case_context(case_dir: Path, dataset_cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    payload = _load_global_payload(case_dir)
    valid = np.ones(len(nodes_df), dtype=bool)
    if bool(dataset_cfg.get("exclude_bc_nodes", False)):
        if "bc_mask" in nodes_df.columns:
            valid &= nodes_df["bc_mask"].to_numpy(dtype=np.float32) <= 0.5
        else:
            valid &= _build_boundary_mask(nodes_df, payload) <= 0.5

    disk_center = _build_disk_center_region_mask(nodes_df) & valid
    earpiece = _build_earpiece_region_mask(
        nodes_df,
        global_payload=payload,
        region_cfg=dataset_cfg.get("earpiece_region"),
    ) & valid
    fullpart = valid
    masks = {
        "disk_center": disk_center.astype(bool, copy=False),
        "earpiece": earpiece.astype(bool, copy=False),
        "fullpart": fullpart.astype(bool, copy=False),
    }
    if "node_index" in nodes_df.columns:
        nodes_for_alignment = nodes_df[["node_index"]].copy()
    else:
        nodes_for_alignment = pd.DataFrame(index=np.arange(len(nodes_df)))
    return masks, nodes_for_alignment


def _load_target_values(path: Path, nodes_for_alignment: pd.DataFrame) -> np.ndarray:
    target_df = pd.read_csv(path, usecols=lambda column: column in {"node_index", "MISES_psd_density"})
    return _load_aligned_target_column(
        target_df,
        target_column=PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_for_alignment,
        target_path=path,
    )


def _sample_region_values(
    sample_path: str,
    masks: dict[str, np.ndarray],
    nodes_for_alignment: pd.DataFrame,
    per_region_cap: int,
) -> dict[str, np.ndarray]:
    path = Path(sample_path)
    target_values = _load_target_values(path, nodes_for_alignment)
    finite = np.isfinite(target_values) & (target_values >= 0.0)
    seed = zlib.crc32(path.as_posix().encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for region_name, mask in masks.items():
        selected = target_values[mask & finite].astype(np.float64, copy=False)
        if selected.size == 0:
            result[region_name] = selected
            continue
        if per_region_cap > 0 and selected.size > per_region_cap:
            indices = rng.choice(selected.size, size=per_region_cap, replace=False)
            selected = selected[indices]
        result[region_name] = selected
    return result


def _count_region_bins(
    sample_path: str,
    masks: dict[str, np.ndarray],
    nodes_for_alignment: pd.DataFrame,
    edges_by_region: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    path = Path(sample_path)
    target_values = _load_target_values(path, nodes_for_alignment)
    finite = np.isfinite(target_values) & (target_values >= 0.0)
    result: dict[str, dict[str, Any]] = {}
    for region_name, mask in masks.items():
        values = target_values[mask & finite].astype(np.float64, copy=False)
        edges = edges_by_region[region_name]
        if values.size == 0:
            result[region_name] = {
                "counts": np.zeros(100, dtype=np.int64),
                "sum": np.zeros(100, dtype=np.float64),
                "min": math.inf,
                "max": -math.inf,
                "points": 0,
            }
            continue
        bin_indices = np.searchsorted(edges, values, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, 99)
        result[region_name] = {
            "counts": np.bincount(bin_indices, minlength=100).astype(np.int64, copy=False),
            "sum": np.bincount(bin_indices, weights=values, minlength=100).astype(np.float64, copy=False),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "points": int(values.size),
        }
    return result


def _scan_region_value_stats(
    sample_path: str,
    masks: dict[str, np.ndarray],
    nodes_for_alignment: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    path = Path(sample_path)
    target_values = _load_target_values(path, nodes_for_alignment)
    finite = np.isfinite(target_values) & (target_values >= 0.0)
    result: dict[str, dict[str, Any]] = {}
    for region_name, mask in masks.items():
        values = target_values[mask & finite].astype(np.float64, copy=False)
        if values.size == 0:
            result[region_name] = {
                "points": 0,
                "positive_points": 0,
                "zero_count": 0,
                "min": math.inf,
                "positive_min": math.inf,
                "max": -math.inf,
            }
            continue
        positive = values[values > 0.0]
        result[region_name] = {
            "points": int(values.size),
            "positive_points": int(positive.size),
            "zero_count": int((values == 0.0).sum()),
            "min": float(np.min(values)),
            "positive_min": float(np.min(positive)) if positive.size else math.inf,
            "max": float(np.max(values)),
        }
    return result


def _count_region_value_histogram(
    sample_path: str,
    masks: dict[str, np.ndarray],
    nodes_for_alignment: pd.DataFrame,
    edges_by_region: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    path = Path(sample_path)
    target_values = _load_target_values(path, nodes_for_alignment)
    finite = np.isfinite(target_values) & (target_values >= 0.0)
    result: dict[str, dict[str, Any]] = {}
    for region_name, mask in masks.items():
        values = target_values[mask & finite].astype(np.float64, copy=False)
        positive = values[values > 0.0]
        edges = edges_by_region[region_name]
        if positive.size == 0:
            result[region_name] = {
                "counts": np.zeros(len(edges) - 1, dtype=np.int64),
                "sum": np.zeros(len(edges) - 1, dtype=np.float64),
            }
            continue
        counts, _ = np.histogram(positive, bins=edges)
        sums, _ = np.histogram(positive, bins=edges, weights=positive)
        result[region_name] = {
            "counts": counts.astype(np.int64, copy=False),
            "sum": sums.astype(np.float64, copy=False),
        }
    return result


def _group_sample_paths_by_case(sample_paths: list[Path]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in sample_paths:
        grouped.setdefault(path.parent.parent.name, []).append(str(path))
    return grouped


def _worker_sample_case(args: tuple[str, list[str], dict[str, Any], int]) -> dict[str, list[np.ndarray]]:
    case_dir_str, sample_paths, dataset_cfg, per_region_cap = args
    masks, nodes_for_alignment = _load_case_context(Path(case_dir_str), dataset_cfg)
    output: dict[str, list[np.ndarray]] = {region: [] for region in REGIONS}
    for sample_path in sample_paths:
        sampled = _sample_region_values(sample_path, masks, nodes_for_alignment, per_region_cap)
        for region_name, values in sampled.items():
            if values.size:
                output[region_name].append(values)
    return output


def _worker_count_case(
    args: tuple[str, list[str], dict[str, Any], dict[str, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    case_dir_str, sample_paths, dataset_cfg, edges_by_region = args
    masks, nodes_for_alignment = _load_case_context(Path(case_dir_str), dataset_cfg)
    output = {
        region_name: {
            "counts": np.zeros(100, dtype=np.int64),
            "sum": np.zeros(100, dtype=np.float64),
            "min": math.inf,
            "max": -math.inf,
            "points": 0,
        }
        for region_name in REGIONS
    }
    for sample_path in sample_paths:
        counted = _count_region_bins(sample_path, masks, nodes_for_alignment, edges_by_region)
        for region_name, stats in counted.items():
            output[region_name]["counts"] += stats["counts"]
            output[region_name]["sum"] += stats["sum"]
            output[region_name]["min"] = min(output[region_name]["min"], stats["min"])
            output[region_name]["max"] = max(output[region_name]["max"], stats["max"])
            output[region_name]["points"] += int(stats["points"])
    return output


def _worker_scan_case(args: tuple[str, list[str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    case_dir_str, sample_paths, dataset_cfg = args
    masks, nodes_for_alignment = _load_case_context(Path(case_dir_str), dataset_cfg)
    output = {
        region_name: {
            "points": 0,
            "positive_points": 0,
            "zero_count": 0,
            "min": math.inf,
            "positive_min": math.inf,
            "max": -math.inf,
        }
        for region_name in REGIONS
    }
    for sample_path in sample_paths:
        scanned = _scan_region_value_stats(sample_path, masks, nodes_for_alignment)
        for region_name, stats in scanned.items():
            output[region_name]["points"] += int(stats["points"])
            output[region_name]["positive_points"] += int(stats["positive_points"])
            output[region_name]["zero_count"] += int(stats["zero_count"])
            output[region_name]["min"] = min(output[region_name]["min"], stats["min"])
            output[region_name]["positive_min"] = min(output[region_name]["positive_min"], stats["positive_min"])
            output[region_name]["max"] = max(output[region_name]["max"], stats["max"])
    return output


def _worker_histogram_case(
    args: tuple[str, list[str], dict[str, Any], dict[str, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    case_dir_str, sample_paths, dataset_cfg, edges_by_region = args
    masks, nodes_for_alignment = _load_case_context(Path(case_dir_str), dataset_cfg)
    output = {
        region_name: {
            "counts": np.zeros(len(edges_by_region[region_name]) - 1, dtype=np.int64),
            "sum": np.zeros(len(edges_by_region[region_name]) - 1, dtype=np.float64),
        }
        for region_name in REGIONS
    }
    for sample_path in sample_paths:
        counted = _count_region_value_histogram(sample_path, masks, nodes_for_alignment, edges_by_region)
        for region_name, stats in counted.items():
            output[region_name]["counts"] += stats["counts"]
            output[region_name]["sum"] += stats["sum"]
    return output


def _consume_with_progress(
    tasks: list[tuple],
    worker_fn: Any,
    workers: int,
    progress_label: str,
    progress_every: int,
) -> list[Any]:
    started = time.monotonic()
    completed = 0
    results: list[Any] = []
    if workers <= 1:
        for task in tasks:
            results.append(worker_fn(task))
            completed += 1
            if completed == 1 or completed == len(tasks) or completed % max(1, progress_every) == 0:
                print(
                    f"{progress_label} | case={completed}/{len(tasks)} | elapsed={_format_duration(time.monotonic() - started)}",
                    flush=True,
                )
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = iter(tasks)
        pending: dict[Any, None] = {}
        max_pending = max(workers, workers * 2)

        def submit_until_full() -> None:
            while len(pending) < max_pending:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
                pending[executor.submit(worker_fn, task)] = None

        submit_until_full()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                results.append(future.result())
                completed += 1
                if completed == 1 or completed == len(tasks) or completed % max(1, progress_every) == 0:
                    print(
                        f"{progress_label} | case={completed}/{len(tasks)} | elapsed={_format_duration(time.monotonic() - started)}",
                        flush=True,
                    )
            submit_until_full()
    return results


def _estimate_edges(
    grouped_paths: dict[str, list[str]],
    dataset_cfg: dict[str, Any],
    sample_values_per_file: int,
    workers: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tasks = [
        (str(Path(dataset_cfg["root"]) / case_name), paths, dataset_cfg, int(sample_values_per_file))
        for case_name, paths in grouped_paths.items()
    ]
    results = _consume_with_progress(
        tasks,
        _worker_sample_case,
        workers=workers,
        progress_label="quantile sampling",
        progress_every=progress_every,
    )
    sampled_by_region: dict[str, list[np.ndarray]] = {region: [] for region in REGIONS}
    for result in results:
        for region_name, arrays in result.items():
            sampled_by_region[region_name].extend(arrays)

    quantiles = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    edges_by_region: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {"sample_values_per_file": int(sample_values_per_file), "regions": {}}
    for region_name, arrays in sampled_by_region.items():
        if not arrays:
            raise ValueError(f"No sampled PSD values for region: {region_name}")
        values = np.concatenate(arrays)
        edges = np.quantile(values, quantiles)
        edges[0] = min(edges[0], float(np.min(values)))
        edges[-1] = max(edges[-1], float(np.max(values)))
        edges_by_region[region_name] = edges.astype(np.float64, copy=False)
        stats["regions"][region_name] = {
            "sampled_values": int(values.size),
            "sample_min": float(np.min(values)),
            "sample_max": float(np.max(values)),
            "sample_mean": float(np.mean(values)),
        }
    return edges_by_region, stats


def _count_bins(
    grouped_paths: dict[str, list[str]],
    dataset_cfg: dict[str, Any],
    edges_by_region: dict[str, np.ndarray],
    workers: int,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    tasks = [
        (str(Path(dataset_cfg["root"]) / case_name), paths, dataset_cfg, edges_by_region)
        for case_name, paths in grouped_paths.items()
    ]
    results = _consume_with_progress(
        tasks,
        _worker_count_case,
        workers=workers,
        progress_label="bin counting",
        progress_every=progress_every,
    )
    accum = {
        region_name: {
            "counts": np.zeros(100, dtype=np.int64),
            "sum": np.zeros(100, dtype=np.float64),
            "min": math.inf,
            "max": -math.inf,
            "points": 0,
        }
        for region_name in REGIONS
    }
    for result in results:
        for region_name, stats in result.items():
            accum[region_name]["counts"] += stats["counts"]
            accum[region_name]["sum"] += stats["sum"]
            accum[region_name]["min"] = min(accum[region_name]["min"], stats["min"])
            accum[region_name]["max"] = max(accum[region_name]["max"], stats["max"])
            accum[region_name]["points"] += int(stats["points"])
    return accum


def _scan_value_ranges(
    grouped_paths: dict[str, list[str]],
    dataset_cfg: dict[str, Any],
    workers: int,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    tasks = [
        (str(Path(dataset_cfg["root"]) / case_name), paths, dataset_cfg)
        for case_name, paths in grouped_paths.items()
    ]
    results = _consume_with_progress(
        tasks,
        _worker_scan_case,
        workers=workers,
        progress_label="value range scan",
        progress_every=progress_every,
    )
    accum = {
        region_name: {
            "points": 0,
            "positive_points": 0,
            "zero_count": 0,
            "min": math.inf,
            "positive_min": math.inf,
            "max": -math.inf,
        }
        for region_name in REGIONS
    }
    for result in results:
        for region_name, stats in result.items():
            accum[region_name]["points"] += int(stats["points"])
            accum[region_name]["positive_points"] += int(stats["positive_points"])
            accum[region_name]["zero_count"] += int(stats["zero_count"])
            accum[region_name]["min"] = min(accum[region_name]["min"], stats["min"])
            accum[region_name]["positive_min"] = min(accum[region_name]["positive_min"], stats["positive_min"])
            accum[region_name]["max"] = max(accum[region_name]["max"], stats["max"])
    return accum


def _build_log_edges(value_stats: dict[str, dict[str, Any]], bins: int) -> dict[str, np.ndarray]:
    edges_by_region: dict[str, np.ndarray] = {}
    for region_name, stats in value_stats.items():
        positive_min = float(stats["positive_min"])
        positive_max = float(stats["max"])
        if not math.isfinite(positive_min) or positive_min <= 0.0:
            positive_min = 1e-12
        if not math.isfinite(positive_max) or positive_max <= positive_min:
            positive_max = positive_min * 10.0
        edges_by_region[region_name] = np.logspace(
            math.log10(positive_min),
            math.log10(positive_max),
            num=max(2, int(bins) + 1),
            dtype=np.float64,
        )
        edges_by_region[region_name][0] = min(edges_by_region[region_name][0], positive_min)
        edges_by_region[region_name][-1] = max(edges_by_region[region_name][-1], positive_max)
    return edges_by_region


def _count_value_histograms(
    grouped_paths: dict[str, list[str]],
    dataset_cfg: dict[str, Any],
    edges_by_region: dict[str, np.ndarray],
    workers: int,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    tasks = [
        (str(Path(dataset_cfg["root"]) / case_name), paths, dataset_cfg, edges_by_region)
        for case_name, paths in grouped_paths.items()
    ]
    results = _consume_with_progress(
        tasks,
        _worker_histogram_case,
        workers=workers,
        progress_label="value histogram counting",
        progress_every=progress_every,
    )
    accum = {
        region_name: {
            "counts": np.zeros(len(edges_by_region[region_name]) - 1, dtype=np.int64),
            "sum": np.zeros(len(edges_by_region[region_name]) - 1, dtype=np.float64),
        }
        for region_name in REGIONS
    }
    for result in results:
        for region_name, stats in result.items():
            accum[region_name]["counts"] += stats["counts"]
            accum[region_name]["sum"] += stats["sum"]
    return accum


def _rows_for_region(region_name: str, edges: np.ndarray, stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = stats["counts"].astype(np.int64, copy=False)
    sums = stats["sum"].astype(np.float64, copy=False)
    total = max(int(counts.sum()), 1)
    for idx in range(100):
        count = int(counts[idx])
        rows.append(
            {
                "region": region_name,
                "region_label": REGIONS[region_name]["label"],
                "quantile_band": f"p{idx}-p{idx + 1}",
                "quantile_start": idx / 100.0,
                "quantile_end": (idx + 1) / 100.0,
                "psd_min": float(edges[idx]),
                "psd_max": float(edges[idx + 1]),
                "count": count,
                "count_fraction": float(count / total),
                "psd_mean": float(sums[idx] / count) if count > 0 else float("nan"),
            }
        )
    return rows


def _histogram_rows_for_region(
    region_name: str,
    edges: np.ndarray,
    hist_stats: dict[str, Any],
    value_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = hist_stats["counts"].astype(np.int64, copy=False)
    sums = hist_stats["sum"].astype(np.float64, copy=False)
    total = max(int(value_stats["points"]), 1)
    positive_total = max(int(value_stats["positive_points"]), 1)
    for idx in range(len(counts)):
        count = int(counts[idx])
        rows.append(
            {
                "region": region_name,
                "region_label": REGIONS[region_name]["label"],
                "bin_index": idx,
                "psd_min": float(edges[idx]),
                "psd_max": float(edges[idx + 1]),
                "count": count,
                "count_fraction_all": float(count / total),
                "count_fraction_positive": float(count / positive_total),
                "psd_mean": float(sums[idx] / count) if count > 0 else float("nan"),
                "zero_count": int(value_stats["zero_count"]) if idx == 0 else 0,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_region(path: Path, rows: list[dict[str, Any]], region_name: str) -> None:
    counts = np.asarray([row["count"] for row in rows], dtype=np.float64)
    left = np.asarray([row["psd_min"] for row in rows], dtype=np.float64)
    right = np.asarray([row["psd_max"] for row in rows], dtype=np.float64)
    finite_positive = np.concatenate([left[left > 0.0], right[right > 0.0]])
    min_positive = float(np.min(finite_positive)) if finite_positive.size else 1e-12
    left = np.where(left > 0.0, left, min_positive * 0.5)
    right = np.maximum(right, left * (1.0 + 1e-6))
    centers = np.sqrt(left * right)
    widths = right - left
    color = str(REGIONS[region_name]["color"])
    label = str(REGIONS[region_name]["label"])

    fig, ax = plt.subplots(figsize=(18, 7.2))
    bars = ax.bar(centers, counts, width=widths, align="center", color=color, alpha=0.84, edgecolor="white", linewidth=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PSD value at one frequency")
    ax.set_ylabel("Count, log scale")
    ax.set_title(f"{label} training PSD distribution by 1% quantile band")
    ax.grid(True, axis="both", which="major", alpha=0.25)
    ax.grid(True, axis="x", which="minor", alpha=0.08)

    for idx in range(0, 100, 5):
        value = counts[idx]
        text = f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value:.0f}"
        ax.annotate(
            f"{rows[idx]['quantile_band']}\n{text}",
            xy=(centers[idx], max(value, 1.0)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
            color="#333333",
        )
    if bars:
        ax.legend([bars[0]], [label], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_combined(path: Path, rows_by_region: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(19, 13.5), sharex=True)
    for ax, region_name in zip(axes, REGIONS):
        rows = rows_by_region[region_name]
        counts = np.asarray([row["count"] for row in rows], dtype=np.float64)
        left = np.asarray([row["psd_min"] for row in rows], dtype=np.float64)
        right = np.asarray([row["psd_max"] for row in rows], dtype=np.float64)
        finite_positive = np.concatenate([left[left > 0.0], right[right > 0.0]])
        min_positive = float(np.min(finite_positive)) if finite_positive.size else 1e-12
        left = np.where(left > 0.0, left, min_positive * 0.5)
        right = np.maximum(right, left * (1.0 + 1e-6))
        centers = np.sqrt(left * right)
        widths = right - left
        color = str(REGIONS[region_name]["color"])
        label = str(REGIONS[region_name]["label"])
        ax.bar(centers, counts, width=widths, align="center", color=color, alpha=0.84, edgecolor="white", linewidth=0.25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Count, log")
        ax.set_title(label, loc="left", fontsize=11)
        ax.grid(True, axis="both", which="major", alpha=0.25)
        ax.grid(True, axis="x", which="minor", alpha=0.08)
        for idx in range(0, 100, 10):
            value = counts[idx]
            text = f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value:.0f}"
            ax.annotate(
                f"p{idx}\n{text}",
                xy=(centers[idx], max(value, 1.0)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7.0,
            )

    axes[-1].set_xlabel("PSD value at one frequency")
    fig.suptitle("Training PSD value distribution by region; bins are 1% quantile intervals", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_histogram_region(
    path: Path,
    rows: list[dict[str, Any]],
    region_name: str,
    zero_count: int,
    total_points: int,
) -> None:
    counts = np.asarray([row["count"] for row in rows], dtype=np.float64)
    left = np.asarray([row["psd_min"] for row in rows], dtype=np.float64)
    right = np.asarray([row["psd_max"] for row in rows], dtype=np.float64)
    centers = np.sqrt(left * right)
    widths = right - left
    color = str(REGIONS[region_name]["color"])
    label = str(REGIONS[region_name]["label"])

    fig, ax = plt.subplots(figsize=(18, 7.2))
    bars = ax.bar(centers, counts, width=widths, align="center", color=color, alpha=0.86, edgecolor="white", linewidth=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PSD value at one frequency")
    ax.set_ylabel("Count, log scale")
    ax.set_title(f"{label} training PSD value histogram, 100 log-spaced bins")
    ax.grid(True, axis="both", which="major", alpha=0.25)
    ax.grid(True, axis="x", which="minor", alpha=0.08)
    zero_ratio = zero_count / max(total_points, 1)
    ax.text(
        0.015,
        0.965,
        f"zero PSD count: {zero_count:,} ({zero_ratio:.3%})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.86},
    )
    if bars:
        ax.legend([bars[0]], [label], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_histogram_combined(
    path: Path,
    rows_by_region: dict[str, list[dict[str, Any]]],
    value_stats: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(19, 13.5), sharex=False)
    for ax, region_name in zip(axes, REGIONS):
        rows = rows_by_region[region_name]
        counts = np.asarray([row["count"] for row in rows], dtype=np.float64)
        left = np.asarray([row["psd_min"] for row in rows], dtype=np.float64)
        right = np.asarray([row["psd_max"] for row in rows], dtype=np.float64)
        centers = np.sqrt(left * right)
        widths = right - left
        color = str(REGIONS[region_name]["color"])
        label = str(REGIONS[region_name]["label"])
        ax.bar(centers, counts, width=widths, align="center", color=color, alpha=0.86, edgecolor="white", linewidth=0.25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Count, log")
        zero_count = int(value_stats[region_name]["zero_count"])
        total_points = int(value_stats[region_name]["points"])
        zero_ratio = zero_count / max(total_points, 1)
        ax.set_title(f"{label} | zero PSD: {zero_count:,} ({zero_ratio:.3%})", loc="left", fontsize=11)
        ax.grid(True, axis="both", which="major", alpha=0.25)
        ax.grid(True, axis="x", which="minor", alpha=0.08)
    axes[-1].set_xlabel("PSD value at one frequency")
    fig.suptitle("Training PSD value histogram by region, 100 log-spaced value bins", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_histogram_mode(
    grouped_paths: dict[str, list[str]],
    dataset_cfg: dict[str, Any],
    output_dir: Path,
    workers: int,
    progress_every: int,
    bins: int,
    combined_plot_name: str,
    summary_base: dict[str, Any],
) -> dict[str, Any]:
    value_stats = _scan_value_ranges(
        grouped_paths=grouped_paths,
        dataset_cfg=dataset_cfg,
        workers=workers,
        progress_every=progress_every,
    )
    edges_by_region = _build_log_edges(value_stats, bins=bins)
    hist_accum = _count_value_histograms(
        grouped_paths=grouped_paths,
        dataset_cfg=dataset_cfg,
        edges_by_region=edges_by_region,
        workers=workers,
        progress_every=progress_every,
    )

    all_rows: list[dict[str, Any]] = []
    rows_by_region: dict[str, list[dict[str, Any]]] = {}
    for region_name in REGIONS:
        rows = _histogram_rows_for_region(region_name, edges_by_region[region_name], hist_accum[region_name], value_stats[region_name])
        rows_by_region[region_name] = rows
        all_rows.extend(rows)
        _write_csv(output_dir / f"{region_name}_train_psd_value_histogram_100bins.csv", rows)
        _plot_histogram_region(
            output_dir / f"{region_name}_train_psd_value_histogram_100bins.png",
            rows,
            region_name,
            zero_count=int(value_stats[region_name]["zero_count"]),
            total_points=int(value_stats[region_name]["points"]),
        )

    _write_csv(output_dir / "train_psd_value_histogram_100bins_all_regions.csv", all_rows)
    _plot_histogram_combined(output_dir / combined_plot_name, rows_by_region, value_stats)

    summary = dict(summary_base)
    summary.update(
        {
            "histogram_bins": int(bins),
            "regions": {
                region_name: {
                    "label": REGIONS[region_name]["label"],
                    "points": int(value_stats[region_name]["points"]),
                    "positive_points": int(value_stats[region_name]["positive_points"]),
                    "zero_count": int(value_stats[region_name]["zero_count"]),
                    "min": None if not math.isfinite(value_stats[region_name]["min"]) else float(value_stats[region_name]["min"]),
                    "positive_min": None
                    if not math.isfinite(value_stats[region_name]["positive_min"])
                    else float(value_stats[region_name]["positive_min"]),
                    "max": None if not math.isfinite(value_stats[region_name]["max"]) else float(value_stats[region_name]["max"]),
                    "histogram_edges": [float(value) for value in edges_by_region[region_name]],
                    "histogram_counted_positive_points": int(hist_accum[region_name]["counts"].sum()),
                }
                for region_name in REGIONS
            },
            "outputs": {
                "combined_plot": str(output_dir / combined_plot_name),
                "combined_csv": str(output_dir / "train_psd_value_histogram_100bins_all_regions.csv"),
                "region_plots": {
                    region_name: str(output_dir / f"{region_name}_train_psd_value_histogram_100bins.png")
                    for region_name in REGIONS
                },
                "region_csvs": {
                    region_name: str(output_dir / f"{region_name}_train_psd_value_histogram_100bins.csv")
                    for region_name in REGIONS
                },
            },
        }
    )
    (output_dir / "train_psd_value_histogram_100bins_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    dataset_cfg = dict(config.get("dataset", {}))
    if not dataset_cfg:
        raise ValueError(f"{args.config} does not contain a dataset section.")
    if args.exclude_bc_nodes is not None:
        dataset_cfg["exclude_bc_nodes"] = bool(args.exclude_bc_nodes)
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    case_dirs = _selected_case_dirs(dataset_cfg, args.split)
    if args.max_cases is not None:
        case_dirs = case_dirs[: int(args.max_cases)]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    grouped_paths = _group_sample_paths_by_case(sample_paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print(
        "train PSD distribution | "
        f"split={args.split} | cases={len(case_dirs)} | frames={len(sample_paths)} | workers={args.num_workers}",
        flush=True,
    )

    summary_base = {
        "config": str(args.config),
        "split": args.split,
        "cases": len(case_dirs),
        "frames": len(sample_paths),
        "exclude_bc_nodes": bool(dataset_cfg.get("exclude_bc_nodes", False)),
    }

    if args.histogram_only:
        summary = _run_histogram_mode(
            grouped_paths=grouped_paths,
            dataset_cfg=dataset_cfg,
            output_dir=output_dir,
            workers=int(args.num_workers),
            progress_every=int(args.progress_every),
            bins=int(args.histogram_bins),
            combined_plot_name=str(args.histogram_combined_plot_name),
            summary_base=summary_base,
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        (output_dir / "train_psd_value_histogram_100bins_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False), flush=True)
        print(f"done | elapsed={_format_duration(time.monotonic() - started)}", flush=True)
        return

    edges_by_region, quantile_stats = _estimate_edges(
        grouped_paths=grouped_paths,
        dataset_cfg=dataset_cfg,
        sample_values_per_file=int(args.sample_values_per_file),
        workers=int(args.num_workers),
        progress_every=int(args.progress_every),
    )
    accum = _count_bins(
        grouped_paths=grouped_paths,
        dataset_cfg=dataset_cfg,
        edges_by_region=edges_by_region,
        workers=int(args.num_workers),
        progress_every=int(args.progress_every),
    )

    all_rows: list[dict[str, Any]] = []
    rows_by_region: dict[str, list[dict[str, Any]]] = {}
    for region_name in REGIONS:
        rows = _rows_for_region(region_name, edges_by_region[region_name], accum[region_name])
        rows_by_region[region_name] = rows
        all_rows.extend(rows)
        _write_csv(output_dir / f"{region_name}_train_psd_p1_distribution.csv", rows)
        _plot_region(output_dir / f"{region_name}_train_psd_p1_distribution.png", rows, region_name)

    _write_csv(output_dir / "train_psd_p1_distribution_all_regions.csv", all_rows)
    _plot_combined(output_dir / args.combined_plot_name, rows_by_region)

    summary = {
        **summary_base,
        "regions": {
            region_name: {
                "label": REGIONS[region_name]["label"],
                "points": int(accum[region_name]["points"]),
                "counted_points": int(accum[region_name]["counts"].sum()),
                "min": None if not math.isfinite(accum[region_name]["min"]) else float(accum[region_name]["min"]),
                "max": None if not math.isfinite(accum[region_name]["max"]) else float(accum[region_name]["max"]),
                "quantile_edges": [float(value) for value in edges_by_region[region_name]],
            }
            for region_name in REGIONS
        },
        "quantile_estimation": quantile_stats,
        "outputs": {
            "combined_plot": str(output_dir / args.combined_plot_name),
            "combined_csv": str(output_dir / "train_psd_p1_distribution_all_regions.csv"),
            "region_plots": {
                region_name: str(output_dir / f"{region_name}_train_psd_p1_distribution.png")
                for region_name in REGIONS
            },
            "region_csvs": {
                region_name: str(output_dir / f"{region_name}_train_psd_p1_distribution.csv")
                for region_name in REGIONS
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "train_psd_p1_distribution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary["outputs"], indent=2, ensure_ascii=False), flush=True)
    print(f"done | elapsed={_format_duration(time.monotonic() - started)}", flush=True)


if __name__ == "__main__":
    main()

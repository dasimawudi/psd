from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from case7_node_mlp.center_curve_low_rank_diagnostics import (
    _build_case_matrix,
    _load_target_stats,
    _select_cases,
    _target_floor,
    _target_floor_needs_stats,
)
from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, write_json


BASIS_FIELDS = [
    "frequency_hz",
    "basis_1",
    "basis_2",
    "basis_3",
    "basis_4",
    "basis_5",
    "singular_value_1",
    "singular_value_2",
    "singular_value_3",
    "singular_value_4",
    "singular_value_5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one global SVD over all selected disk-center target curves.")
    parser.add_argument("--config", type=str, default="node/configs/node_mlp_v6_disk_center_hotspot_features.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=str, default="node/outputs/node_mlp_v6_disk_center_global_low_rank_train_all")
    parser.add_argument("--num-cases", type=int, default=0, help="0 means all cases in split.")
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--target-floor", type=str, default=None)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--grid", choices=["log", "linear"], default="log")
    return parser.parse_args()


def _frequency_grid(dataset_cfg: dict[str, Any], grid: str, grid_size: int) -> np.ndarray:
    minimum = float(dataset_cfg.get("min_frequency_hz", 20.0))
    maximum = float(dataset_cfg.get("max_frequency_hz", 2000.0))
    size = max(2, int(grid_size))
    if grid == "log":
        return np.geomspace(max(minimum, 1e-6), maximum, num=size, dtype=np.float64)
    return np.linspace(minimum, maximum, num=size, dtype=np.float64)


def _interpolate_matrix(matrix: np.ndarray, frequencies: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(frequencies)
    frequencies = np.asarray(frequencies[order], dtype=np.float64)
    matrix = np.asarray(matrix[:, order], dtype=np.float64)
    unique_frequencies, unique_indices = np.unique(frequencies, return_index=True)
    frequencies = unique_frequencies
    matrix = matrix[:, unique_indices]
    if frequencies.size == 1:
        return np.repeat(matrix[:, :1], grid.size, axis=1)

    right = np.searchsorted(frequencies, grid, side="right")
    left = np.clip(right - 1, 0, frequencies.size - 2)
    right = left + 1
    denom = np.maximum(frequencies[right] - frequencies[left], 1e-12)
    weight = ((grid - frequencies[left]) / denom).reshape(1, -1)
    return matrix[:, left] * (1.0 - weight) + matrix[:, right] * weight


def _rank_explained(eigenvalues: np.ndarray, rank: int) -> float:
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(eigenvalues[: min(int(rank), eigenvalues.size)].sum() / total)


def _plot_rank_explained(path: Path, ranks: list[int], explained: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, explained, marker="o", linewidth=2.2, color="tab:red")
    ax.set_xlabel("rank")
    ax.set_ylabel("explained energy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.suptitle("Global disk-center target curve low-rank structure")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_basis(path: Path, grid: np.ndarray, eigenvectors: np.ndarray, singular_values: np.ndarray, curve_count: int) -> None:
    component_count = min(5, eigenvectors.shape[1])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for idx in range(component_count):
        basis = eigenvectors[:, idx]
        axes[0].plot(grid, basis, linewidth=1.7, label=f"basis {idx + 1}")
        axes[1].plot(
            grid,
            singular_values[idx] * basis / np.sqrt(max(int(curve_count), 1)),
            linewidth=1.7,
            label=f"S{idx + 1}/sqrt(N) * basis {idx + 1}",
        )
    axes[0].set_ylabel("Global right singular vector")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Frequency Hz")
    axes[1].set_ylabel("Typical log contribution")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle("Global SVD frequency basis")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_basis_csv(path: Path, grid: np.ndarray, eigenvectors: np.ndarray, singular_values: np.ndarray) -> None:
    component_count = min(5, eigenvectors.shape[1])
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=BASIS_FIELDS)
        writer.writeheader()
        for freq_index, frequency in enumerate(grid):
            row: dict[str, float] = {"frequency_hz": float(frequency)}
            for idx in range(5):
                row[f"basis_{idx + 1}"] = float(eigenvectors[freq_index, idx]) if idx < component_count else float("nan")
                row[f"singular_value_{idx + 1}"] = float(singular_values[idx]) if idx < singular_values.size else float("nan")
            writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    logger = make_logger(output_dir, logger_name="case7_node_mlp.global_center_curve_low_rank_diagnostics", log_file="global_low_rank_diagnostics.log")
    dataset_cfg = dict(config.get("dataset", {}))
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)
    case_index = discover_case_index(dataset_cfg["root"])
    splits = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    selected_case_names = _select_cases(
        list(splits.get(args.split, [])),
        args.case_name,
        num_cases=int(args.num_cases),
        seed=int(args.seed),
    )
    selected_case_dirs = [case_index[name] for name in selected_case_names]
    all_sample_paths = expand_case_sample_paths(selected_case_dirs, dataset_cfg)
    target_stats = _load_target_stats(all_sample_paths, dataset_cfg) if _target_floor_needs_stats(config, args.target_floor) else {}
    floor = _target_floor(config, target_stats, args.target_floor)
    grid = _frequency_grid(dataset_cfg, args.grid, int(args.grid_size))

    curve_count = 0
    case_count = 0
    column_sum = np.zeros(grid.size, dtype=np.float64)
    gram = np.zeros((grid.size, grid.size), dtype=np.float64)
    case_rows: list[dict[str, Any]] = []
    for offset, case_dir in enumerate(selected_case_dirs, start=1):
        sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
        logger.info("Case %s/%s | %s | frames=%s", offset, len(selected_case_dirs), case_dir.name, len(sample_paths))
        matrix, frequencies, _positions, _radii, _top1_positions, _top5_positions = _build_case_matrix(
            case_dir=case_dir,
            sample_paths=sample_paths,
            dataset_cfg=dataset_cfg,
            target_floor=floor,
            max_nodes=int(args.max_nodes),
            top_fraction=float(args.top_fraction),
        )
        interpolated = _interpolate_matrix(matrix, frequencies, grid)
        row_centered = interpolated - interpolated.mean(axis=1, keepdims=True)
        case_curves = int(row_centered.shape[0])
        curve_count += case_curves
        case_count += 1
        column_sum += row_centered.sum(axis=0)
        gram += row_centered.T @ row_centered
        case_rows.append(
            {
                "case_name": case_dir.name,
                "frequencies": int(len(frequencies)),
                "curves": case_curves,
            }
        )

    if curve_count <= 0:
        raise RuntimeError("No curves were accumulated for global SVD.")
    column_mean = column_sum / float(curve_count)
    centered_gram = gram - float(curve_count) * np.outer(column_mean, column_mean)
    centered_gram = 0.5 * (centered_gram + centered_gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(centered_gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    for idx in range(min(5, eigenvectors.shape[1])):
        anchor = int(np.argmax(np.abs(eigenvectors[:, idx])))
        if float(eigenvectors[anchor, idx]) < 0.0:
            eigenvectors[:, idx] *= -1.0
    singular_values = np.sqrt(eigenvalues)
    ranks = [1, 2, 3, 5, 10, 20, 50]
    explained = {f"rank{rank}_explained": _rank_explained(eigenvalues, rank) for rank in ranks}

    case_csv = output_dir / f"{args.split}_global_svd_cases.csv"
    with case_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["case_name", "frequencies", "curves"])
        writer.writeheader()
        writer.writerows(case_rows)
    basis_csv = output_dir / f"{args.split}_global_svd_basis.csv"
    basis_plot = output_dir / f"{args.split}_global_svd_basis.png"
    rank_plot = output_dir / f"{args.split}_global_rank_explained.png"
    _write_basis_csv(basis_csv, grid, eigenvectors, singular_values)
    _plot_basis(basis_plot, grid, eigenvectors, singular_values, curve_count)
    _plot_rank_explained(rank_plot, ranks, [explained[f"rank{rank}_explained"] for rank in ranks])

    summary = {
        "config": str(args.config),
        "split": args.split,
        "case_count": int(case_count),
        "curve_count": int(curve_count),
        "target_floor": float(floor),
        "max_nodes_per_case": int(args.max_nodes),
        "grid": args.grid,
        "grid_size": int(grid.size),
        "frequency_min_hz": float(grid.min()),
        "frequency_max_hz": float(grid.max()),
        **explained,
        "singular_values_top10": [float(value) for value in singular_values[:10]],
        "basis_csv": str(basis_csv),
        "case_csv": str(case_csv),
        "basis_plot": str(basis_plot),
        "rank_plot": str(rank_plot),
    }
    write_json(output_dir / f"{args.split}_global_low_rank_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

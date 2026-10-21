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
)
from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, write_json


CURVE_FIELDS = [
    "split",
    "case_name",
    "frequency_hz",
    "basis_1",
    "basis_2",
    "basis_3",
    "component_1_log_rms_units",
    "component_2_log_rms_units",
    "component_3_log_rms_units",
    "frequency_common_log_offset",
    "singular_value_1",
    "singular_value_2",
    "singular_value_3",
    "rank1_explained",
    "rank2_explained",
    "rank3_explained",
]

SUMMARY_FIELDS = [
    "split",
    "case_name",
    "frequencies",
    "nodes",
    "rank1_explained",
    "rank2_explained",
    "rank3_explained",
    "rank5_explained",
    "rank10_explained",
    "singular_value_1",
    "singular_value_2",
    "singular_value_3",
    "basis_plot",
    "reconstruction_plot",
    "case_curve_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot explicit SVD rank-3 frequency basis curves from target data.")
    parser.add_argument("--config", type=str, default="node/configs/node_mlp_v6_disk_center_hotspot_features.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=str, default="node/outputs/node_mlp_v6_disk_center_low_rank_svd_curves_train")
    parser.add_argument("--num-cases", type=int, default=32)
    parser.add_argument("--case-name", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--target-floor", type=str, default=None)
    parser.add_argument("--reconstruction-nodes", type=int, default=6)
    return parser.parse_args()


def _rank_explained(singular_values: np.ndarray, rank: int) -> float:
    energy = singular_values * singular_values
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(energy[: min(int(rank), energy.size)].sum() / total)


def _svd_decomposition(matrix: np.ndarray) -> dict[str, np.ndarray | float]:
    row_mean = matrix.mean(axis=1, keepdims=True)
    row_centered = matrix - row_mean
    frequency_mean = row_centered.mean(axis=0, keepdims=True)
    centered = row_centered - frequency_mean
    if min(centered.shape) <= 1 or float(np.linalg.norm(centered)) <= 1e-12:
        raise ValueError("Cannot run SVD on an empty or near-constant matrix.")

    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    component_count = min(3, vt.shape[0])
    # SVD signs are arbitrary. Align each basis so its largest-magnitude point is positive.
    for component in range(component_count):
        anchor = int(np.argmax(np.abs(vt[component])))
        if float(vt[component, anchor]) < 0.0:
            vt[component] *= -1.0
            u[:, component] *= -1.0
    return {
        "row_mean": row_mean,
        "frequency_mean": frequency_mean,
        "centered": centered,
        "u": u,
        "singular_values": singular_values,
        "vt": vt,
    }


def _component_log_rms_units(singular_value: float, basis: np.ndarray, node_count: int) -> np.ndarray:
    # Vt is unitless/unit-norm. Multiplying by S/sqrt(nodes) gives a typical log-space
    # contribution magnitude for a node with RMS-sized coefficient.
    return float(singular_value) * basis / np.sqrt(max(int(node_count), 1))


def _plot_case_basis(path: Path, case_name: str, frequencies: np.ndarray, vt: np.ndarray, singular_values: np.ndarray, node_count: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for component in range(3):
        axes[0].plot(
            frequencies,
            vt[component],
            label=f"basis {component + 1}",
            linewidth=1.8,
            color=colors[component],
        )
        axes[1].plot(
            frequencies,
            _component_log_rms_units(float(singular_values[component]), vt[component], node_count),
            label=f"S{component + 1}/sqrt(N) * basis {component + 1}",
            linewidth=1.8,
            color=colors[component],
        )
    axes[0].set_ylabel("Right singular vector")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Frequency Hz")
    axes[1].set_ylabel("Typical log contribution")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"{case_name} | explicit SVD rank-3 frequency basis")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _select_reconstruction_rows(matrix: np.ndarray, count: int) -> list[int]:
    node_peak = matrix.max(axis=1)
    order = np.argsort(node_peak)
    if order.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, max(int(count), 1))
    rows = {int(order[min(order.size - 1, max(0, int(round(q * (order.size - 1)))))] ) for q in quantiles}
    rows.add(int(order[-1]))
    return sorted(rows, key=lambda row: float(node_peak[row]), reverse=True)[: max(int(count), 1)]


def _plot_reconstruction(
    path: Path,
    case_name: str,
    frequencies: np.ndarray,
    matrix: np.ndarray,
    row_mean: np.ndarray,
    frequency_mean: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    rows: list[int],
) -> None:
    rank = min(3, vt.shape[0])
    centered_rank3 = (u[:, :rank] * singular_values[:rank]) @ vt[:rank]
    reconstructed = centered_rank3 + frequency_mean + row_mean
    if not rows:
        rows = _select_reconstruction_rows(matrix, 6)
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(3.0, 2.2 * len(rows))), sharex=True)
    axes_array = np.atleast_1d(axes)
    node_peak = matrix.max(axis=1)
    for ax, row in zip(axes_array, rows):
        ax.plot(frequencies, matrix[row], label="target log1p", linewidth=1.8)
        ax.plot(frequencies, reconstructed[row], label="rank3 reconstruction", linewidth=1.5, linestyle="--")
        ax.set_ylabel(f"row {row}\npeak={node_peak[row]:.2f}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes_array[-1].set_xlabel("Frequency Hz")
    fig.suptitle(f"{case_name} | target curves reconstructed by top-3 SVD components")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_rank_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    ranks = [1, 2, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary_rows:
        ax.plot(ranks, [float(row[f"rank{rank}_explained"]) for rank in ranks], color="tab:blue", alpha=0.18)
    medians = [float(np.nanmedian([float(row[f"rank{rank}_explained"]) for row in summary_rows])) for rank in ranks]
    means = [float(np.nanmean([float(row[f"rank{rank}_explained"]) for row in summary_rows])) for rank in ranks]
    ax.plot(ranks, medians, marker="o", linewidth=2.2, color="tab:red", label="median")
    ax.plot(ranks, means, marker="s", linewidth=2.0, color="tab:orange", label="mean")
    ax.set_xlabel("rank")
    ax.set_ylabel("explained energy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.suptitle("SVD low-rank explained energy")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_basis_overlay(path: Path, all_curve_rows: list[dict[str, Any]], *, value_prefix: str, ylabel: str, title: str) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in all_curve_rows:
        by_case.setdefault(str(row["case_name"]), []).append(row)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for component, ax in enumerate(axes, start=1):
        value_key = f"{value_prefix}_{component}"
        if value_prefix == "component":
            value_key = f"component_{component}_log_rms_units"
        for case_name, rows in by_case.items():
            ordered = sorted(rows, key=lambda item: float(item["frequency_hz"]))
            ax.plot(
                [float(item["frequency_hz"]) for item in ordered],
                [float(item[value_key]) for item in ordered],
                alpha=0.22,
                linewidth=1.1,
            )
        ax.set_ylabel(f"{ylabel} {component}")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Frequency Hz")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    cases_dir = ensure_dir(output_dir / "cases")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.plot_low_rank_svd_curves", log_file="plot_low_rank_svd_curves.log")
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
    target_stats = _load_target_stats(all_sample_paths, dataset_cfg)
    floor = _target_floor(config, target_stats, args.target_floor)

    all_curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
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
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        matrix = matrix[:, order]

        svd = _svd_decomposition(matrix)
        row_mean = np.asarray(svd["row_mean"])
        frequency_mean = np.asarray(svd["frequency_mean"])
        u = np.asarray(svd["u"])
        singular_values = np.asarray(svd["singular_values"])
        vt = np.asarray(svd["vt"])
        if vt.shape[0] < 3:
            raise RuntimeError(f"Case {case_dir.name} has fewer than 3 SVD components.")

        case_curve_rows: list[dict[str, Any]] = []
        explained = {rank: _rank_explained(singular_values, rank) for rank in [1, 2, 3, 5, 10]}
        for freq_index, frequency in enumerate(frequencies):
            row = {
                "split": args.split,
                "case_name": case_dir.name,
                "frequency_hz": float(frequency),
                "basis_1": float(vt[0, freq_index]),
                "basis_2": float(vt[1, freq_index]),
                "basis_3": float(vt[2, freq_index]),
                "component_1_log_rms_units": float(_component_log_rms_units(singular_values[0], vt[0], matrix.shape[0])[freq_index]),
                "component_2_log_rms_units": float(_component_log_rms_units(singular_values[1], vt[1], matrix.shape[0])[freq_index]),
                "component_3_log_rms_units": float(_component_log_rms_units(singular_values[2], vt[2], matrix.shape[0])[freq_index]),
                "frequency_common_log_offset": float(frequency_mean.reshape(-1)[freq_index]),
                "singular_value_1": float(singular_values[0]),
                "singular_value_2": float(singular_values[1]),
                "singular_value_3": float(singular_values[2]),
                "rank1_explained": float(explained[1]),
                "rank2_explained": float(explained[2]),
                "rank3_explained": float(explained[3]),
            }
            case_curve_rows.append(row)
            all_curve_rows.append(row)

        stem = case_dir.name
        case_curve_csv = cases_dir / f"{stem}_svd_rank3_curves.csv"
        basis_plot = cases_dir / f"{stem}_svd_rank3_basis.png"
        reconstruction_plot = cases_dir / f"{stem}_svd_rank3_reconstruction.png"
        _write_csv(case_curve_csv, case_curve_rows, CURVE_FIELDS)
        _plot_case_basis(basis_plot, case_dir.name, frequencies, vt, singular_values, matrix.shape[0])
        reconstruction_rows = _select_reconstruction_rows(matrix, int(args.reconstruction_nodes))
        _plot_reconstruction(
            reconstruction_plot,
            case_dir.name,
            frequencies,
            matrix,
            row_mean,
            frequency_mean,
            u,
            singular_values,
            vt,
            reconstruction_rows,
        )

        summary_rows.append(
            {
                "split": args.split,
                "case_name": case_dir.name,
                "frequencies": int(frequencies.size),
                "nodes": int(matrix.shape[0]),
                "rank1_explained": float(explained[1]),
                "rank2_explained": float(explained[2]),
                "rank3_explained": float(explained[3]),
                "rank5_explained": float(explained[5]),
                "rank10_explained": float(explained[10]),
                "singular_value_1": float(singular_values[0]),
                "singular_value_2": float(singular_values[1]),
                "singular_value_3": float(singular_values[2]),
                "basis_plot": str(basis_plot),
                "reconstruction_plot": str(reconstruction_plot),
                "case_curve_csv": str(case_curve_csv),
            }
        )

    curves_csv = output_dir / f"{args.split}_svd_rank3_curves_all_cases.csv"
    summary_csv = output_dir / f"{args.split}_svd_rank3_summary.csv"
    _write_csv(curves_csv, all_curve_rows, CURVE_FIELDS)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    rank_plot = output_dir / f"{args.split}_rank_explained.png"
    basis_overlay = output_dir / f"{args.split}_svd_rank3_basis_overlay.png"
    component_overlay = output_dir / f"{args.split}_svd_rank3_log_component_overlay.png"
    _plot_rank_summary(rank_plot, summary_rows)
    _plot_basis_overlay(
        basis_overlay,
        all_curve_rows,
        value_prefix="basis",
        ylabel="basis",
        title=f"{args.split} split | SVD rank-3 basis overlay",
    )
    _plot_basis_overlay(
        component_overlay,
        all_curve_rows,
        value_prefix="component",
        ylabel="log RMS component",
        title=f"{args.split} split | SVD rank-3 typical log contribution overlay",
    )

    aggregate: dict[str, Any] = {
        "config": str(args.config),
        "split": args.split,
        "cases": len(summary_rows),
        "target_floor": float(floor),
        "max_nodes": int(args.max_nodes),
        "curves_csv": str(curves_csv),
        "summary_csv": str(summary_csv),
        "rank_explained_plot": str(rank_plot),
        "basis_overlay_plot": str(basis_overlay),
        "log_component_overlay_plot": str(component_overlay),
        "cases_dir": str(cases_dir),
    }
    for field in ["rank1_explained", "rank2_explained", "rank3_explained", "rank5_explained", "rank10_explained"]:
        values = np.asarray([float(row[field]) for row in summary_rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else float("nan")
        aggregate[f"{field}_median"] = float(np.median(values)) if values.size else float("nan")
    write_json(output_dir / f"{args.split}_svd_rank3_summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

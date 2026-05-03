from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from case7_gnn.data import discover_case_index, resolve_case_splits
from case7_gnn.runtime import ensure_dir, read_config, write_json


DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
TARGET_COLUMNS = ("RTA", "RMises")
TARGET_THRESHOLDS = {
    "RTA": (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0),
    "RMises": (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0),
}


def format_threshold(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "")


def sample_case_values(values: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or values.size == 0:
        return np.empty(0, dtype=np.float64)
    if values.size <= sample_size:
        return values.astype(np.float64, copy=True)
    indices = rng.choice(values.size, size=sample_size, replace=False)
    return values[indices].astype(np.float64, copy=False)


def sample_case_rows(frame: pd.DataFrame, sample_size: int, rng: np.random.Generator) -> pd.DataFrame:
    if sample_size <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= sample_size:
        return frame.copy()
    indices = rng.choice(len(frame), size=sample_size, replace=False)
    return frame.iloc[np.sort(indices)].copy()


@dataclass
class RunningValueStats:
    thresholds: tuple[float, ...]
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    zero_count: int = 0
    positive_count: int = 0
    threshold_counts: dict[float, int] = field(init=False)

    def __post_init__(self) -> None:
        self.threshold_counts = {float(threshold): 0 for threshold in self.thresholds}

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return

        self.count += int(flat.size)
        self.total += float(flat.sum())
        self.total_sq += float(np.square(flat).sum())
        self.min_value = min(self.min_value, float(flat.min()))
        self.max_value = max(self.max_value, float(flat.max()))
        self.zero_count += int((flat == 0.0).sum())
        self.positive_count += int((flat > 0.0).sum())
        for threshold in self.thresholds:
            self.threshold_counts[float(threshold)] += int((flat <= threshold).sum())

    def to_summary(self, quantiles: dict[str, float]) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "cv": None,
                "min": None,
                "max": None,
                "zero_ratio": None,
                "positive_ratio": None,
                "threshold_ratios": {},
                "quantiles": quantiles,
            }

        mean = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean * mean), 0.0)
        std = math.sqrt(variance)
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "cv": (std / mean) if mean != 0.0 else None,
            "min": self.min_value,
            "max": self.max_value,
            "zero_ratio": self.zero_count / self.count,
            "positive_ratio": self.positive_count / self.count,
            "threshold_ratios": {
                f"le_{format_threshold(threshold)}": self.threshold_counts[threshold] / self.count
                for threshold in self.thresholds
            },
            "quantiles": quantiles,
        }


def quantiles_from_sample(values: np.ndarray, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> dict[str, float]:
    if values.size == 0:
        return {}
    computed = np.quantile(values, quantiles)
    return {f"q{int(q * 1000):03d}": float(v) for q, v in zip(quantiles, computed)}


def build_target_stats() -> dict[str, RunningValueStats]:
    return {target: RunningValueStats(thresholds=TARGET_THRESHOLDS[target]) for target in TARGET_COLUMNS}


def read_node_targets(case_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(case_dir / "nodes.csv", usecols=["bc_mask", *TARGET_COLUMNS], dtype="string")
    for column in TARGET_COLUMNS:
        cleaned = frame[column].str.replace(r"\s+", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.isna().any():
            examples = cleaned[numeric.isna()].head(5).tolist()
            raise ValueError(f"Failed to parse {column} values in {case_dir}: {examples}")
        frame[column] = numeric.astype(np.float64)
    frame["bc_mask"] = pd.to_numeric(frame["bc_mask"], errors="coerce").fillna(0).astype(np.int8)
    return frame[["bc_mask", *TARGET_COLUMNS]]


def compute_case_quantiles(values: np.ndarray) -> dict[str, float]:
    quantile_values = np.quantile(values, DEFAULT_QUANTILES)
    return {
        "p50": float(quantile_values[0]),
        "p90": float(quantile_values[1]),
        "p95": float(quantile_values[2]),
        "p99": float(quantile_values[3]),
        "p999": float(quantile_values[4]),
    }


def sample_correlation(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {"pearson_raw": None, "pearson_log1p": None, "spearman_raw": None}
    rta = frame["RTA"].to_numpy(dtype=np.float64, copy=False)
    rmises = frame["RMises"].to_numpy(dtype=np.float64, copy=False)
    if len(rta) < 2:
        return {"pearson_raw": None, "pearson_log1p": None, "spearman_raw": None}
    pearson_raw = float(np.corrcoef(rta, rmises)[0, 1])
    pearson_log = float(np.corrcoef(np.log1p(rta), np.log1p(rmises))[0, 1])
    spearman = float(frame[["RTA", "RMises"]].corr(method="spearman").iloc[0, 1])
    return {
        "pearson_raw": pearson_raw,
        "pearson_log1p": pearson_log,
        "spearman_raw": spearman,
    }


def sample_joint_tail(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {}
    rta_q99 = float(frame["RTA"].quantile(0.99))
    rmises_q99 = float(frame["RMises"].quantile(0.99))
    rta_hot = frame["RTA"] >= rta_q99
    rmises_hot = frame["RMises"] >= rmises_q99
    both_hot = rta_hot & rmises_hot
    return {
        "rta_q99": rta_q99,
        "rmises_q99": rmises_q99,
        "rta_hot_ratio": float(rta_hot.mean()),
        "rmises_hot_ratio": float(rmises_hot.mean()),
        "both_hot_ratio": float(both_hot.mean()),
        "rmises_hot_given_rta_hot": float(both_hot.sum() / max(int(rta_hot.sum()), 1)),
        "rta_hot_given_rmises_hot": float(both_hot.sum() / max(int(rmises_hot.sum()), 1)),
    }


def top_case_rows(case_df: pd.DataFrame, target: str, metric: str, limit: int = 10) -> list[dict[str, Any]]:
    column = f"{target}_{metric}"
    records = case_df.sort_values(column, ascending=False).head(limit).to_dict(orient="records")
    return records


def append_target_report(lines: list[str], target: str, summary: dict[str, Any]) -> None:
    global_stats = summary["global"][target]
    lines.extend(
        [
            f"## {target}",
            "",
            "### Global",
            "",
            "- Metric source: count/mean/std/cv/min/max/zero ratio are full-data exact; quantiles are sample-based approximate.",
            f"- Nodes: {global_stats['count']}",
            f"- Mean / Std: {global_stats['mean']:.6g} / {global_stats['std']:.6g}",
            f"- CV: {global_stats['cv']:.6g}" if global_stats["cv"] is not None else "- CV: n/a",
            f"- Min / Max: {global_stats['min']:.6g} / {global_stats['max']:.6g}",
            f"- q50 / q90 / q95 / q99 / q999: {global_stats['quantiles'].get('q500', float('nan')):.6g} / {global_stats['quantiles'].get('q900', float('nan')):.6g} / {global_stats['quantiles'].get('q950', float('nan')):.6g} / {global_stats['quantiles'].get('q990', float('nan')):.6g} / {global_stats['quantiles'].get('q999', float('nan')):.6g}",
            f"- Zero ratio: {global_stats['zero_ratio']:.6%}",
            "",
            "### Split Stats",
            "",
        ]
    )

    for split_name in ("train", "val", "test"):
        split_stats = summary["splits"][split_name][target]
        lines.extend(
            [
                f"#### {split_name}",
                "",
                "- Metric source: count/mean/std are full-data exact; quantiles are sample-based approximate.",
                f"- Nodes: {split_stats['count']}",
                f"- Mean / Std: {split_stats['mean']:.6g} / {split_stats['std']:.6g}",
                f"- q90 / q95 / q99 / q999: {split_stats['quantiles'].get('q900', float('nan')):.6g} / {split_stats['quantiles'].get('q950', float('nan')):.6g} / {split_stats['quantiles'].get('q990', float('nan')):.6g} / {split_stats['quantiles'].get('q999', float('nan')):.6g}",
                "",
            ]
        )

    lines.extend(["### bc_mask Groups", ""])
    for group_name in ("free_nodes", "fixed_nodes"):
        group_stats = summary["bc_groups"][group_name][target]
        lines.extend(
            [
                f"#### {group_name}",
                "",
                "- Metric source: count/mean/std are full-data exact; quantiles are sample-based approximate.",
                f"- Nodes: {group_stats['count']}",
                f"- Mean / Std: {group_stats['mean']:.6g} / {group_stats['std']:.6g}",
                f"- q90 / q95 / q99: {group_stats['quantiles'].get('q900', float('nan')):.6g} / {group_stats['quantiles'].get('q950', float('nan')):.6g} / {group_stats['quantiles'].get('q990', float('nan')):.6g}",
                "",
            ]
        )


def append_case_tables(lines: list[str], title: str, rows: list[dict[str, Any]], target: str) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "| case | split | mean | p99 | max | nodes |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_name']} | {row['split']} | {row[f'{target}_mean']:.6g} | {row[f'{target}_p99']:.6g} | {row[f'{target}_max']:.6g} | {row['num_nodes']} |"
        )
    lines.append("")


def write_markdown_report(path: Path, summary: dict[str, Any], case_df: pd.DataFrame) -> None:
    lines = [
        "# Node Target Distribution Report",
        "",
        f"- Dataset root: `{summary['dataset']['root']}`",
        f"- Cases: {summary['dataset']['num_cases']}",
        f"- Nodes: {summary['dataset']['total_nodes']}",
        f"- Split sizes: train={summary['dataset']['split_sizes']['train']}, val={summary['dataset']['split_sizes']['val']}, test={summary['dataset']['split_sizes']['test']}",
        f"- Sampling strategy: {summary['sampling']['sampling_strategy']}",
        "",
        "## Metric Provenance",
        "",
        "- Full-data exact metrics: counts, mean, std, cv, min, max, zero ratio, threshold ratios, and all per-case statistics in the case-level CSV.",
        "- Sample-based approximate metrics: all global/split/bc-group quantiles (`q50/q90/q95/q99/q999`), sample correlation, and sampled hotspot-overlap metrics.",
        "- Mixed metrics: `cases_case_p99_above_global_q99` uses exact per-case `p99` values, but the referenced global `q99` is sample-based.",
        "",
        "## Executive Summary",
        "",
        f"- RTA global q99 is {summary['global']['RTA']['quantiles'].get('q990', float('nan')):.6g}, while RMises global q99 is {summary['global']['RMises']['quantiles'].get('q990', float('nan')):.6g}.",
        f"- RTA mean/median gap: {summary['global']['RTA']['mean']:.6g} vs {summary['global']['RTA']['quantiles'].get('q500', float('nan')):.6g}, indicating {'a strong long tail' if summary['global']['RTA']['mean'] > 3 * summary['global']['RTA']['quantiles'].get('q500', 1.0) else 'a moderate tail'}.",
        f"- RMises mean/median gap: {summary['global']['RMises']['mean']:.6g} vs {summary['global']['RMises']['quantiles'].get('q500', float('nan')):.6g}, indicating {'a strong long tail' if summary['global']['RMises']['mean'] > 3 * summary['global']['RMises']['quantiles'].get('q500', 1.0) else 'a moderate tail'}.",
        f"- Sample correlation between RTA and RMises: raw Pearson={summary['joint']['sample_correlation']['pearson_raw']:.6g}, log1p Pearson={summary['joint']['sample_correlation']['pearson_log1p']:.6g}, Spearman={summary['joint']['sample_correlation']['spearman_raw']:.6g}.",
        f"- Q99 hotspot overlap on the sampled nodes: both-hot ratio={summary['joint']['sample_joint_tail']['both_hot_ratio']:.6%}, RMises-hot given RTA-hot={summary['joint']['sample_joint_tail']['rmises_hot_given_rta_hot']:.6%}.",
        "",
    ]

    append_target_report(lines, "RTA", summary)
    append_target_report(lines, "RMises", summary)

    lines.extend(["## Joint Analysis", ""])
    lines.append("- Metric source: sample-based approximate.")
    for key, value in summary["joint"]["sample_correlation"].items():
        lines.append(f"- {key}: {value:.6g}" if value is not None else f"- {key}: n/a")
    for key, value in summary["joint"]["sample_joint_tail"].items():
        lines.append(f"- {key}: {value:.6g}" if value is not None else f"- {key}: n/a")
    lines.append("")

    lines.extend(["## Case-Level Extremes", ""])
    lines.append("- Metric source: full-data exact within each case.")
    lines.append("")
    for target in TARGET_COLUMNS:
        append_case_tables(lines, f"Top {target} Cases By p99", summary["case_level"][target]["top_by_p99"], target)
        append_case_tables(lines, f"Top {target} Cases By max", summary["case_level"][target]["top_by_max"], target)

    lines.extend(["## Case-Level Aggregates", ""])
    for target in TARGET_COLUMNS:
        aggregate = summary["case_level"][target]["aggregate"]
        lines.extend(
            [
                f"### {target}",
                "",
                "- Metric source: case means and case p99 values are exact per-case statistics; the comparison against global q99 uses the sample-based global q99.",
                f"- Mean case mean: {aggregate['mean_case_mean']:.6g}",
                f"- Median case mean: {aggregate['median_case_mean']:.6g}",
                f"- Mean case p99: {aggregate['mean_case_p99']:.6g}",
                f"- Median case p99: {aggregate['median_case_p99']:.6g}",
                f"- Cases with p99 > q99(global): {aggregate['cases_case_p99_above_global_q99']}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze node-level RTA and RMises distributions across the case7 dataset.")
    parser.add_argument("--config", type=str, default="configs/field.yaml", help="Field config used for dataset split.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/node_target_distribution",
        help="Directory for reports and tables.",
    )
    parser.add_argument("--sample-size", type=int, default=600000, help="Approximate total sample size for target quantiles and joint stats.")
    parser.add_argument("--split-sample-size", type=int, default=180000, help="Approximate total sample size per split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case-level sampling.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N cases.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = read_config(args.config)
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"])

    output_dir = ensure_dir(args.output_dir)
    available_cases = discover_case_index(dataset_root)
    splits = resolve_case_splits(dataset_root, dataset_cfg)
    case_to_split = {case_name: split_name for split_name, names in splits.items() for case_name in names}
    total_cases = len(available_cases)

    global_stats = build_target_stats()
    split_stats = {split_name: build_target_stats() for split_name in splits}
    bc_group_stats = {group_name: build_target_stats() for group_name in ("free_nodes", "fixed_nodes")}

    global_rng = np.random.default_rng(args.seed)
    joint_rng = np.random.default_rng(args.seed + 1)
    split_rngs = {split_name: np.random.default_rng(args.seed + 100 + idx) for idx, split_name in enumerate(splits)}
    per_case_sample = max(1, math.ceil(args.sample_size / max(total_cases, 1)))
    per_split_case_sample = {
        split_name: max(1, math.ceil(args.split_sample_size / max(len(names), 1)))
        for split_name, names in splits.items()
    }

    sample_parts = {target: [] for target in TARGET_COLUMNS}
    split_sample_parts = {split_name: {target: [] for target in TARGET_COLUMNS} for split_name in splits}
    bc_group_sample_parts = {
        group_name: {target: [] for target in TARGET_COLUMNS}
        for group_name in ("free_nodes", "fixed_nodes")
    }
    joint_sample_parts: list[pd.DataFrame] = []
    case_rows: list[dict[str, Any]] = []
    total_nodes = 0

    for case_index, case_name in enumerate(sorted(available_cases), start=1):
        case_dir = available_cases[case_name]
        split_name = case_to_split.get(case_name, "unspecified")
        frame = read_node_targets(case_dir)
        total_nodes += len(frame)

        free_mask = frame["bc_mask"] == 0
        fixed_mask = ~free_mask

        for target in TARGET_COLUMNS:
            values = frame[target].to_numpy(dtype=np.float64, copy=False)
            free_values = frame.loc[free_mask, target].to_numpy(dtype=np.float64, copy=False)
            fixed_values = frame.loc[fixed_mask, target].to_numpy(dtype=np.float64, copy=False)
            global_stats[target].update(values)
            sample_parts[target].append(sample_case_values(values, per_case_sample, global_rng))
            if split_name in split_stats:
                split_stats[split_name][target].update(values)
                split_sample_parts[split_name][target].append(
                    sample_case_values(values, per_split_case_sample[split_name], split_rngs[split_name])
                )
            bc_group_stats["free_nodes"][target].update(free_values)
            bc_group_stats["fixed_nodes"][target].update(fixed_values)
            bc_group_sample_parts["free_nodes"][target].append(sample_case_values(free_values, per_case_sample, global_rng))
            bc_group_sample_parts["fixed_nodes"][target].append(sample_case_values(fixed_values, per_case_sample, global_rng))

        joint_sample_parts.append(sample_case_rows(frame[["RTA", "RMises"]], per_case_sample, joint_rng))

        row: dict[str, Any] = {
            "case_name": case_name,
            "split": split_name,
            "num_nodes": int(len(frame)),
            "fixed_node_ratio": float(fixed_mask.mean()),
        }
        for target in TARGET_COLUMNS:
            values = frame[target].to_numpy(dtype=np.float64, copy=False)
            q = compute_case_quantiles(values)
            row[f"{target}_mean"] = float(values.mean())
            row[f"{target}_std"] = float(values.std())
            row[f"{target}_max"] = float(values.max())
            for quantile_name, quantile_value in q.items():
                row[f"{target}_{quantile_name}"] = quantile_value
        case_rows.append(row)

        if args.progress_every > 0 and (case_index % args.progress_every == 0 or case_index == total_cases):
            print(f"[{case_index}/{total_cases}] processed {case_name}")

    sampled = {
        target: (np.concatenate(parts) if parts else np.empty(0, dtype=np.float64))
        for target, parts in sample_parts.items()
    }
    split_sampled = {
        split_name: {
            target: (np.concatenate(parts) if parts else np.empty(0, dtype=np.float64))
            for target, parts in target_parts.items()
        }
        for split_name, target_parts in split_sample_parts.items()
    }
    bc_group_sampled = {
        group_name: {
            target: (np.concatenate(parts) if parts else np.empty(0, dtype=np.float64))
            for target, parts in target_parts.items()
        }
        for group_name, target_parts in bc_group_sample_parts.items()
    }
    joint_sample = pd.concat(joint_sample_parts, ignore_index=True) if joint_sample_parts else pd.DataFrame(columns=["RTA", "RMises"])

    summary = {
        "dataset": {
            "root": str(dataset_root),
            "config_path": str(Path(args.config)),
            "output_dir": str(output_dir),
            "num_cases": total_cases,
            "total_nodes": total_nodes,
            "split_sizes": {split_name: len(names) for split_name, names in splits.items()},
        },
        "sampling": {
            "sampling_strategy": "fixed random sample per case",
            "per_case_sample_size": per_case_sample,
            "per_split_case_sample_size": per_split_case_sample,
            "global_sample_size": {target: int(values.size) for target, values in sampled.items()},
            "split_sample_size": {
                split_name: {target: int(values.size) for target, values in target_parts.items()}
                for split_name, target_parts in split_sampled.items()
            },
            "joint_sample_size": int(len(joint_sample)),
            "quantiles_are_approximate": True,
        },
        "global": {
            target: global_stats[target].to_summary(quantiles_from_sample(sampled[target]))
            for target in TARGET_COLUMNS
        },
        "splits": {
            split_name: {
                target: split_stats[split_name][target].to_summary(quantiles_from_sample(split_sampled[split_name][target]))
                for target in TARGET_COLUMNS
            }
            for split_name in splits
        },
        "bc_groups": {
            group_name: {
                target: bc_group_stats[group_name][target].to_summary(quantiles_from_sample(bc_group_sampled[group_name][target]))
                for target in TARGET_COLUMNS
            }
            for group_name in ("free_nodes", "fixed_nodes")
        },
        "joint": {
            "sample_correlation": sample_correlation(joint_sample),
            "sample_joint_tail": sample_joint_tail(joint_sample),
        },
    }

    case_df = pd.DataFrame(case_rows).sort_values("case_name").reset_index(drop=True)
    case_df.to_csv(output_dir / "case_level_node_target_summary.csv", index=False, encoding="utf-8")

    global_q99 = {
        target: summary["global"][target]["quantiles"].get("q990", math.inf)
        for target in TARGET_COLUMNS
    }
    summary["case_level"] = {}
    for target in TARGET_COLUMNS:
        summary["case_level"][target] = {
            "top_by_p99": top_case_rows(case_df, target, "p99"),
            "top_by_max": top_case_rows(case_df, target, "max"),
            "aggregate": {
                "mean_case_mean": float(case_df[f"{target}_mean"].mean()),
                "median_case_mean": float(case_df[f"{target}_mean"].median()),
                "mean_case_p99": float(case_df[f"{target}_p99"].mean()),
                "median_case_p99": float(case_df[f"{target}_p99"].median()),
                "cases_case_p99_above_global_q99": int((case_df[f"{target}_p99"] > global_q99[target]).sum()),
            },
        }

    write_json(output_dir / "node_target_distribution_summary.json", summary)
    write_markdown_report(output_dir / "node_target_distribution_report.md", summary, case_df)

    print(f"Summary written to: {output_dir / 'node_target_distribution_summary.json'}")
    print(f"Case-level CSV written to: {output_dir / 'case_level_node_target_summary.csv'}")
    print(f"Report written to: {output_dir / 'node_target_distribution_report.md'}")


if __name__ == "__main__":
    main()

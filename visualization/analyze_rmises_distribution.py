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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None


DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
DEFAULT_THRESHOLDS = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def sample_case_values(values: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size <= 0 or values.size == 0:
        return np.empty(0, dtype=np.float64)
    if values.size <= sample_size:
        return values.astype(np.float64, copy=True)
    indices = rng.choice(values.size, size=sample_size, replace=False)
    return values[indices].astype(np.float64, copy=False)


@dataclass
class RunningValueStats:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    negative_count: int = 0
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
        self.negative_count += int((flat < 0.0).sum())
        self.zero_count += int((flat == 0.0).sum())
        self.positive_count += int((flat > 0.0).sum())

        for threshold in self.thresholds:
            self.threshold_counts[float(threshold)] += int((flat <= threshold).sum())

    def to_summary(self, quantile_values: dict[str, float] | None = None) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "negative_count": 0,
                "negative_ratio": None,
                "zero_count": 0,
                "zero_ratio": None,
                "positive_count": 0,
                "positive_ratio": None,
                "threshold_ratios": {},
                "quantiles": quantile_values or {},
            }

        mean = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean * mean), 0.0)
        threshold_ratios = {
            f"le_{format_threshold(threshold)}": self.threshold_counts[threshold] / self.count
            for threshold in self.thresholds
        }
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min_value,
            "max": self.max_value,
            "negative_count": self.negative_count,
            "negative_ratio": self.negative_count / self.count,
            "zero_count": self.zero_count,
            "zero_ratio": self.zero_count / self.count,
            "positive_count": self.positive_count,
            "positive_ratio": self.positive_count / self.count,
            "threshold_ratios": threshold_ratios,
            "quantiles": quantile_values or {},
        }


def format_threshold(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "")


def read_rmises(case_dir: Path) -> np.ndarray:
    series = pd.read_csv(case_dir / "nodes.csv", usecols=["RMises"], dtype={"RMises": "string"})["RMises"]
    cleaned = series.str.replace(r"\s+", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.isna().any():
        bad_examples = cleaned[numeric.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse RMises values in {case_dir}: {bad_examples}")
    return numeric.to_numpy(dtype=np.float64, copy=False)


def quantiles_from_sample(values: np.ndarray, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> dict[str, float]:
    if values.size == 0:
        return {}
    computed = np.quantile(values, quantiles)
    return {f"q{int(q * 1000):03d}": float(v) for q, v in zip(quantiles, computed)}


def top_case_rows(case_rows: list[dict[str, Any]], column: str, limit: int = 10) -> list[dict[str, Any]]:
    return sorted(case_rows, key=lambda row: float(row[column]), reverse=True)[:limit]


def create_summary_figure(
    output_path: Path,
    raw_sample: np.ndarray,
    clamped_sample: np.ndarray,
    split_samples: dict[str, np.ndarray],
    case_df: pd.DataFrame,
    split_stats: dict[str, dict[str, Any]],
) -> str | None:
    if plt is None:
        return "matplotlib is unavailable; skipped figure generation."

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if raw_sample.size > 0:
        raw_upper = float(np.quantile(raw_sample, 0.995))
        raw_lower = float(min(raw_sample.min(), 0.0))
        axes[0, 0].hist(raw_sample, bins=120, range=(raw_lower, raw_upper), color="#3b82f6", alpha=0.85)
        axes[0, 0].axvline(0.0, color="#ef4444", linewidth=1.5, linestyle="--")
        axes[0, 0].set_title("Raw RMises sample (clipped at 99.5%)")
        axes[0, 0].set_xlabel("RMises")
        axes[0, 0].set_ylabel("Sample count")
    else:
        axes[0, 0].set_axis_off()

    if clamped_sample.size > 0:
        axes[0, 1].hist(np.log1p(clamped_sample), bins=120, color="#f59e0b", alpha=0.85)
        axes[0, 1].set_title("Clamped RMises sample in log1p space")
        axes[0, 1].set_xlabel("log1p(RMises)")
        axes[0, 1].set_ylabel("Sample count")
    else:
        axes[0, 1].set_axis_off()

    split_labels: list[str] = []
    split_box_data: list[np.ndarray] = []
    for split_name in ("train", "val", "test"):
        sample = split_samples.get(split_name)
        if sample is None or sample.size == 0:
            continue
        split_labels.append(split_name)
        split_box_data.append(np.log1p(sample))
    if split_box_data:
        axes[1, 0].boxplot(split_box_data, labels=split_labels, showfliers=False)
        axes[1, 0].set_title("Split comparison in log1p space")
        axes[1, 0].set_ylabel("log1p(RMises)")

        summary_lines = []
        for split_name in split_labels:
            summary = split_stats[split_name]["clamped"]
            q99 = summary["quantiles"].get("q990")
            summary_lines.append(
                f"{split_name}: mean={summary['mean']:.4g}, q99={q99:.4g}" if q99 is not None else f"{split_name}: no data"
            )
        axes[1, 0].text(
            0.03,
            0.97,
            "\n".join(summary_lines),
            transform=axes[1, 0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
        )
    else:
        axes[1, 0].set_axis_off()

    positive_case_df = case_df[(case_df["clamped_mean"] > 0.0) & (case_df["clamped_p99"] > 0.0)]
    if not positive_case_df.empty:
        scatter = axes[1, 1].scatter(
            positive_case_df["clamped_mean"],
            positive_case_df["clamped_p99"],
            c=positive_case_df["negative_ratio"],
            cmap="viridis",
            s=16,
            alpha=0.75,
        )
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Case mean vs case p99")
        axes[1, 1].set_xlabel("Case mean RMises (clamped)")
        axes[1, 1].set_ylabel("Case p99 RMises (clamped)")
        colorbar = fig.colorbar(scatter, ax=axes[1, 1])
        colorbar.set_label("Negative ratio in raw RMises")
    else:
        axes[1, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return None


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    top_max_cases: list[dict[str, Any]],
    top_p99_cases: list[dict[str, Any]],
    top_negative_cases: list[dict[str, Any]],
) -> None:
    raw = summary["global"]["raw"]
    clamped = summary["global"]["clamped"]
    lines = [
        "# RMises Distribution Report",
        "",
        f"- Cases: {summary['dataset']['num_cases']}",
        f"- Nodes: {summary['dataset']['total_nodes']}",
        f"- Clamp negative RMises for training: {summary['dataset']['clamp_negative_rmises']}",
        "",
        "## Global",
        "",
        f"- Raw mean: {raw['mean']:.6g}",
        f"- Raw std: {raw['std']:.6g}",
        f"- Raw min/max: {raw['min']:.6g} / {raw['max']:.6g}",
        f"- Raw negative ratio: {raw['negative_ratio']:.6%}",
        f"- Clamped zero ratio: {clamped['zero_ratio']:.6%}",
        f"- Clamped q90/q95/q99: {clamped['quantiles'].get('q900', float('nan')):.6g} / {clamped['quantiles'].get('q950', float('nan')):.6g} / {clamped['quantiles'].get('q990', float('nan')):.6g}",
        "",
        "## Split Stats",
        "",
    ]

    for split_name in ("train", "val", "test"):
        if split_name not in summary["splits"]:
            continue
        split_clamped = summary["splits"][split_name]["clamped"]
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Nodes: {split_clamped['count']}",
                f"- Mean: {split_clamped['mean']:.6g}",
                f"- Std: {split_clamped['std']:.6g}",
                f"- q90/q95/q99: {split_clamped['quantiles'].get('q900', float('nan')):.6g} / {split_clamped['quantiles'].get('q950', float('nan')):.6g} / {split_clamped['quantiles'].get('q990', float('nan')):.6g}",
                "",
            ]
        )

    def append_case_table(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend([f"## {title}", "", "| case | split | mean | p99 | max | negative_ratio |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(
                f"| {row['case_name']} | {row['split']} | {row['clamped_mean']:.6g} | {row['clamped_p99']:.6g} | {row['clamped_max']:.6g} | {row['negative_ratio']:.6%} |"
            )
        lines.append("")

    append_case_table("Top Cases By Max", top_max_cases)
    append_case_table("Top Cases By P99", top_p99_cases)
    append_case_table("Top Cases By Negative Ratio", top_negative_cases)

    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the raw RMises distribution across the case7 dataset.")
    parser.add_argument("--config", type=str, default="configs/field.yaml", help="Field config used for dataset split and clamp settings.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/rmises_distribution",
        help="Directory for reports and figures.",
    )
    parser.add_argument("--sample-size", type=int, default=500000, help="Approximate total sample size for global quantiles.")
    parser.add_argument("--split-sample-size", type=int, default=150000, help="Approximate total sample size per split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for case-level sampling.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N cases.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = read_config(args.config)
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"])
    clamp_negative_rmises = bool(dataset_cfg.get("clamp_negative_rmises", True))

    output_dir = ensure_dir(args.output_dir)
    available_cases = discover_case_index(dataset_root)
    splits = resolve_case_splits(dataset_root, dataset_cfg)
    case_to_split = {
        case_name: split_name
        for split_name, names in splits.items()
        for case_name in names
    }
    total_cases = len(available_cases)

    global_raw = RunningValueStats()
    global_clamped = RunningValueStats()
    split_raw = {name: RunningValueStats() for name in splits}
    split_clamped = {name: RunningValueStats() for name in splits}

    global_rng = np.random.default_rng(args.seed)
    split_rngs = {
        split_name: np.random.default_rng(args.seed + 100 + idx)
        for idx, split_name in enumerate(splits)
    }
    per_case_sample = max(1, math.ceil(args.sample_size / max(total_cases, 1)))
    per_split_case_sample = {
        split_name: max(1, math.ceil(args.split_sample_size / max(len(names), 1)))
        for split_name, names in splits.items()
    }
    raw_sample_parts: list[np.ndarray] = []
    clamped_sample_parts: list[np.ndarray] = []
    split_raw_sample_parts: dict[str, list[np.ndarray]] = {split_name: [] for split_name in splits}
    split_clamped_sample_parts: dict[str, list[np.ndarray]] = {split_name: [] for split_name in splits}

    case_rows: list[dict[str, Any]] = []
    total_nodes = 0

    for case_index, case_name in enumerate(sorted(available_cases), start=1):
        case_dir = available_cases[case_name]
        split_name = case_to_split.get(case_name, "unspecified")
        raw_values = read_rmises(case_dir)
        clamped_values = np.maximum(raw_values, 0.0) if clamp_negative_rmises else raw_values

        total_nodes += int(raw_values.size)
        global_raw.update(raw_values)
        global_clamped.update(clamped_values)
        raw_sample_parts.append(sample_case_values(raw_values, per_case_sample, global_rng))
        clamped_sample_parts.append(sample_case_values(clamped_values, per_case_sample, global_rng))

        if split_name in split_raw:
            split_raw[split_name].update(raw_values)
            split_clamped[split_name].update(clamped_values)
            split_raw_sample_parts[split_name].append(
                sample_case_values(raw_values, per_split_case_sample[split_name], split_rngs[split_name])
            )
            split_clamped_sample_parts[split_name].append(
                sample_case_values(clamped_values, per_split_case_sample[split_name], split_rngs[split_name])
            )

        quantile_values = np.quantile(clamped_values, DEFAULT_QUANTILES)
        case_rows.append(
            {
                "case_name": case_name,
                "split": split_name,
                "num_nodes": int(raw_values.size),
                "raw_min": float(raw_values.min()),
                "raw_max": float(raw_values.max()),
                "raw_mean": float(raw_values.mean()),
                "raw_std": float(raw_values.std()),
                "clamped_mean": float(clamped_values.mean()),
                "clamped_std": float(clamped_values.std()),
                "clamped_p50": float(quantile_values[0]),
                "clamped_p90": float(quantile_values[1]),
                "clamped_p95": float(quantile_values[2]),
                "clamped_p99": float(quantile_values[3]),
                "clamped_p999": float(quantile_values[4]),
                "clamped_max": float(clamped_values.max()),
                "negative_count": int((raw_values < 0.0).sum()),
                "negative_ratio": float((raw_values < 0.0).mean()),
                "zero_ratio_raw": float((raw_values == 0.0).mean()),
                "zero_ratio_clamped": float((clamped_values == 0.0).mean()),
            }
        )

        if args.progress_every > 0 and (case_index % args.progress_every == 0 or case_index == total_cases):
            print(f"[{case_index}/{total_cases}] processed {case_name}")

    raw_sample = np.concatenate(raw_sample_parts) if raw_sample_parts else np.empty(0, dtype=np.float64)
    clamped_sample = np.concatenate(clamped_sample_parts) if clamped_sample_parts else np.empty(0, dtype=np.float64)
    split_raw_samples = {
        split_name: (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        )
        for split_name, parts in split_raw_sample_parts.items()
    }
    split_samples = {
        split_name: (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        )
        for split_name, parts in split_clamped_sample_parts.items()
    }

    summary = {
        "dataset": {
            "root": str(dataset_root),
            "config_path": str(Path(args.config)),
            "output_dir": str(output_dir),
            "num_cases": total_cases,
            "total_nodes": total_nodes,
            "clamp_negative_rmises": clamp_negative_rmises,
            "split_sizes": {split_name: len(names) for split_name, names in splits.items()},
        },
        "sampling": {
            "global_sample_size": int(raw_sample.size),
            "split_sample_sizes": {split_name: int(values.size) for split_name, values in split_samples.items()},
            "quantiles_are_approximate": True,
            "sampling_strategy": "fixed random sample per case",
            "per_case_sample_size": per_case_sample,
            "per_split_case_sample_size": per_split_case_sample,
        },
        "global": {
            "raw": global_raw.to_summary(quantile_values=quantiles_from_sample(raw_sample)),
            "clamped": global_clamped.to_summary(quantile_values=quantiles_from_sample(clamped_sample)),
        },
        "splits": {
            split_name: {
                "raw": split_raw[split_name].to_summary(quantile_values=quantiles_from_sample(split_raw_samples[split_name])),
                "clamped": split_clamped[split_name].to_summary(quantile_values=quantiles_from_sample(split_samples[split_name])),
            }
            for split_name in splits
        },
    }

    case_df = pd.DataFrame(case_rows).sort_values("case_name").reset_index(drop=True)
    case_df.to_csv(output_dir / "case_level_rmises_summary.csv", index=False, encoding="utf-8")

    top_max_cases = top_case_rows(case_rows, "clamped_max")
    top_p99_cases = top_case_rows(case_rows, "clamped_p99")
    top_negative_cases = top_case_rows(case_rows, "negative_ratio")
    summary["case_level"] = {
        "top_by_max": top_max_cases,
        "top_by_p99": top_p99_cases,
        "top_by_negative_ratio": top_negative_cases,
        "aggregate": {
            "mean_case_mean": float(case_df["clamped_mean"].mean()),
            "median_case_mean": float(case_df["clamped_mean"].median()),
            "mean_case_p99": float(case_df["clamped_p99"].mean()),
            "median_case_p99": float(case_df["clamped_p99"].median()),
            "mean_negative_ratio": float(case_df["negative_ratio"].mean()),
        },
    }

    figure_note = create_summary_figure(
        output_path=output_dir / "rmises_distribution_summary.png",
        raw_sample=raw_sample,
        clamped_sample=clamped_sample,
        split_samples=split_samples,
        case_df=case_df,
        split_stats=summary["splits"],
    )
    if figure_note is not None:
        summary["figure_note"] = figure_note

    write_json(output_dir / "rmises_distribution_summary.json", summary)
    write_markdown_report(
        path=output_dir / "rmises_distribution_report.md",
        summary=summary,
        top_max_cases=top_max_cases,
        top_p99_cases=top_p99_cases,
        top_negative_cases=top_negative_cases,
    )

    print(f"Summary written to: {output_dir / 'rmises_distribution_summary.json'}")
    print(f"Case-level CSV written to: {output_dir / 'case_level_rmises_summary.csv'}")
    if figure_note is None:
        print(f"Figure written to: {output_dir / 'rmises_distribution_summary.png'}")
    else:
        print(figure_note)


if __name__ == "__main__":
    main()

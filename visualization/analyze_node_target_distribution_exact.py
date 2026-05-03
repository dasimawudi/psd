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
SPLIT_NAMES = ("train", "val", "test")
BC_GROUPS = ("free_nodes", "fixed_nodes")
TARGET_THRESHOLDS = {
    "RTA": (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0),
    "RMises": (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0),
}


def format_threshold(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}".replace("+", "")


def read_node_targets(case_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(case_dir / "nodes.csv", usecols=["bc_mask", *TARGET_COLUMNS], dtype="string")
    for column in TARGET_COLUMNS:
        cleaned = frame[column].str.replace(r"\s+", "", regex=True)
        cleaned = cleaned.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.isna().any():
            examples = cleaned[numeric.isna()].head(5).tolist()
            raise ValueError(f"Failed to parse {column} values in {case_dir}: {examples}")
        frame[column] = numeric.astype(np.float64)
    frame["bc_mask"] = pd.to_numeric(frame["bc_mask"], errors="coerce").fillna(0).astype(np.int8)
    return frame[["bc_mask", *TARGET_COLUMNS]]


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
        variance = max((self.total_sq / self.count) - mean * mean, 0.0)
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


@dataclass
class RunningPairStats:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        if xv.size == 0:
            return
        self.count += int(xv.size)
        self.sum_x += float(xv.sum())
        self.sum_y += float(yv.sum())
        self.sum_x2 += float(np.square(xv).sum())
        self.sum_y2 += float(np.square(yv).sum())
        self.sum_xy += float((xv * yv).sum())

    def pearson(self) -> float | None:
        if self.count <= 1:
            return None
        numerator = (self.count * self.sum_xy) - (self.sum_x * self.sum_y)
        left = (self.count * self.sum_x2) - (self.sum_x * self.sum_x)
        right = (self.count * self.sum_y2) - (self.sum_y * self.sum_y)
        denominator = math.sqrt(max(left, 0.0) * max(right, 0.0))
        if denominator == 0.0:
            return None
        return numerator / denominator


def build_target_stats() -> dict[str, RunningValueStats]:
    return {target: RunningValueStats(thresholds=TARGET_THRESHOLDS[target]) for target in TARGET_COLUMNS}


def compute_case_quantiles(values: np.ndarray) -> dict[str, float]:
    quantile_values = np.quantile(values, DEFAULT_QUANTILES)
    return {
        "p50": float(quantile_values[0]),
        "p90": float(quantile_values[1]),
        "p95": float(quantile_values[2]),
        "p99": float(quantile_values[3]),
        "p999": float(quantile_values[4]),
    }


def exact_quantiles_from_sorted(values: np.memmap | np.ndarray, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> dict[str, float]:
    size = int(values.shape[0])
    if size == 0:
        return {}

    result: dict[str, float] = {}
    for quantile in quantiles:
        position = quantile * (size - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        lower_value = float(values[lower])
        upper_value = float(values[upper])
        if lower == upper:
            value = lower_value
        else:
            value = lower_value + (upper_value - lower_value) * (position - lower)
        result[f"q{int(quantile * 1000):03d}"] = value
    return result


def build_scope_paths(temp_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        target: {
            "global": temp_dir / f"{target.lower()}__global.dat",
            **{split_name: temp_dir / f"{target.lower()}__{split_name}.dat" for split_name in SPLIT_NAMES},
            **{group_name: temp_dir / f"{target.lower()}__{group_name}.dat" for group_name in BC_GROUPS},
        }
        for target in TARGET_COLUMNS
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 节点目标分布分析报告（严格全量精确分位数版）",
        "",
        f"- 数据集路径：`{summary['dataset']['root']}`",
        f"- case 数量：{summary['dataset']['num_cases']}",
        f"- 节点总数：{summary['dataset']['total_nodes']}",
        f"- 数据划分：train={summary['dataset']['split_sizes']['train']}，val={summary['dataset']['split_sizes']['val']}，test={summary['dataset']['split_sizes']['test']}",
        f"- 分位数口径：{summary['quantile_method']}",
        f"- 联合统计口径：{summary['joint']['method']}",
        "",
        "## 一、执行摘要",
        "",
        f"- `RTA` 的全局 `q99` 为 `{summary['global']['RTA']['quantiles'].get('q990', float('nan')):.6g}`，`RMises` 的全局 `q99` 为 `{summary['global']['RMises']['quantiles'].get('q990', float('nan')):.6g}`。",
        f"- `RTA` 的均值与中位数分别为 `{summary['global']['RTA']['mean']:.6g}` 和 `{summary['global']['RTA']['quantiles'].get('q500', float('nan')):.6g}`，整体分布较集中。",
        f"- `RMises` 的均值与中位数分别为 `{summary['global']['RMises']['mean']:.6g}` 和 `{summary['global']['RMises']['quantiles'].get('q500', float('nan')):.6g}`，属于强长尾分布。",
        f"- `RTA` 与 `RMises` 的全量 Pearson 相关系数：raw={summary['joint']['pearson_raw']:.6g}，log1p={summary['joint']['pearson_log1p']:.6g}。",
        f"- 按全量精确 q99 阈值统计，`RTA` 热点与 `RMises` 热点的重叠比例为 `{summary['joint']['both_hot_ratio']:.6%}`。",
        "",
    ]

    for target in TARGET_COLUMNS:
        global_stats = summary["global"][target]
        lines.extend(
            [
                f"## 二、{target} 分布分析" if target == "RTA" else f"## 三、{target} 分布分析",
                "",
                "### 全局分布",
                "",
                f"- 节点数：{global_stats['count']}",
                f"- 均值 / 标准差：`{global_stats['mean']:.6g} / {global_stats['std']:.6g}`",
                f"- 变异系数 CV：`{global_stats['cv']:.6g}`" if global_stats["cv"] is not None else "- 变异系数 CV：n/a",
                f"- 最小值 / 最大值：`{global_stats['min']:.6g} / {global_stats['max']:.6g}`",
                f"- 分位数：`q50={global_stats['quantiles'].get('q500', float('nan')):.6g}`，`q90={global_stats['quantiles'].get('q900', float('nan')):.6g}`，`q95={global_stats['quantiles'].get('q950', float('nan')):.6g}`，`q99={global_stats['quantiles'].get('q990', float('nan')):.6g}`，`q999={global_stats['quantiles'].get('q999', float('nan')):.6g}`",
                f"- 零值占比：`{global_stats['zero_ratio']:.6%}`",
                "",
                "### 按数据划分比较",
                "",
            ]
        )

        for split_name in SPLIT_NAMES:
            split_stats = summary["splits"][split_name][target]
            lines.extend(
                [
                    f"#### {split_name}",
                    "",
                    f"- 节点数：{split_stats['count']}",
                    f"- 均值 / 标准差：`{split_stats['mean']:.6g} / {split_stats['std']:.6g}`",
                    f"- 分位数：`q90={split_stats['quantiles'].get('q900', float('nan')):.6g}`，`q95={split_stats['quantiles'].get('q950', float('nan')):.6g}`，`q99={split_stats['quantiles'].get('q990', float('nan')):.6g}`，`q999={split_stats['quantiles'].get('q999', float('nan')):.6g}`",
                    "",
                ]
            )

        lines.extend(["### 按 bc_mask 分组", ""])
        for group_name in BC_GROUPS:
            group_stats = summary["bc_groups"][group_name][target]
            lines.extend(
                [
                    f"#### {group_name}",
                    "",
                    f"- 节点数：{group_stats['count']}",
                    f"- 均值 / 标准差：`{group_stats['mean']:.6g} / {group_stats['std']:.6g}`",
                    f"- 分位数：`q90={group_stats['quantiles'].get('q900', float('nan')):.6g}`，`q95={group_stats['quantiles'].get('q950', float('nan')):.6g}`，`q99={group_stats['quantiles'].get('q990', float('nan')):.6g}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## 四、RTA 与 RMises 联合分析",
            "",
            f"- 全量 Pearson（原始值）：`{summary['joint']['pearson_raw']:.6g}`",
            f"- 全量 Pearson（log1p）：`{summary['joint']['pearson_log1p']:.6g}`",
            f"- 全量 q99 阈值：`RTA={summary['joint']['rta_q99']:.6g}`，`RMises={summary['joint']['rmises_q99']:.6g}`",
            f"- `RTA` 热点占比：`{summary['joint']['rta_hot_ratio']:.6%}`",
            f"- `RMises` 热点占比：`{summary['joint']['rmises_hot_ratio']:.6%}`",
            f"- 同时为热点的占比：`{summary['joint']['both_hot_ratio']:.6%}`",
            f"- 在 `RTA` 热点中同时也是 `RMises` 热点的比例：`{summary['joint']['rmises_hot_given_rta_hot']:.6%}`",
            f"- 在 `RMises` 热点中同时也是 `RTA` 热点的比例：`{summary['joint']['rta_hot_given_rmises_hot']:.6%}`",
            "",
            "## 五、Case 级极值分析",
            "",
        ]
    )

    for target in TARGET_COLUMNS:
        lines.extend(
            [
                f"### Top {target} Cases By p99",
                "",
                "| case | split | mean | p99 | max | nodes |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["case_level"][target]["top_by_p99"]:
            lines.append(
                f"| {row['case_name']} | {row['split']} | {row[f'{target}_mean']:.6g} | {row[f'{target}_p99']:.6g} | {row[f'{target}_max']:.6g} | {row['num_nodes']} |"
            )
        lines.extend(["", f"### Top {target} Cases By max", "", "| case | split | mean | p99 | max | nodes |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in summary["case_level"][target]["top_by_max"]:
            lines.append(
                f"| {row['case_name']} | {row['split']} | {row[f'{target}_mean']:.6g} | {row[f'{target}_p99']:.6g} | {row[f'{target}_max']:.6g} | {row['num_nodes']} |"
            )
        lines.append("")

    lines.extend(["## 六、Case 级整体统计", ""])
    for target in TARGET_COLUMNS:
        aggregate = summary["case_level"][target]["aggregate"]
        lines.extend(
            [
                f"### {target}",
                "",
                f"- case 均值的平均值：`{aggregate['mean_case_mean']:.6g}`",
                f"- case 均值的中位数：`{aggregate['median_case_mean']:.6g}`",
                f"- case p99 的平均值：`{aggregate['mean_case_p99']:.6g}`",
                f"- case p99 的中位数：`{aggregate['median_case_p99']:.6g}`",
                f"- `p99 > 全局 q99` 的 case 数：`{aggregate['cases_case_p99_above_global_q99']}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze node-level RTA and RMises distributions with exact full quantiles.")
    parser.add_argument("--config", type=str, default="configs/field.yaml", help="Field config used for dataset split.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/node_target_distribution_exact",
        help="Directory for exact reports and tables.",
    )
    parser.add_argument("--temp-dir", type=str, default=".cache/node_target_distribution_exact", help="Directory for temporary memmap files.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N cases.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = read_config(args.config)
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_cfg["root"])

    output_dir = ensure_dir(args.output_dir)
    temp_dir = ensure_dir(args.temp_dir)
    available_cases = discover_case_index(dataset_root)
    splits = resolve_case_splits(dataset_root, dataset_cfg)
    case_to_split = {case_name: split_name for split_name, names in splits.items() for case_name in names}
    selected_case_names = sorted(case_to_split)
    total_cases = len(selected_case_names)

    global_stats = build_target_stats()
    split_stats = {split_name: build_target_stats() for split_name in SPLIT_NAMES}
    bc_group_stats = {group_name: build_target_stats() for group_name in BC_GROUPS}
    pair_stats_raw = RunningPairStats()
    pair_stats_log = RunningPairStats()
    case_rows: list[dict[str, Any]] = []

    counts = {
        target: {
            "global": 0,
            **{split_name: 0 for split_name in SPLIT_NAMES},
            **{group_name: 0 for group_name in BC_GROUPS},
        }
        for target in TARGET_COLUMNS
    }

    total_nodes = 0
    for case_index, case_name in enumerate(selected_case_names, start=1):
        frame = read_node_targets(available_cases[case_name])
        split_name = case_to_split.get(case_name, "unspecified")
        free_mask = frame["bc_mask"] == 0
        fixed_mask = ~free_mask
        node_count = int(len(frame))
        total_nodes += node_count

        row: dict[str, Any] = {
            "case_name": case_name,
            "split": split_name,
            "num_nodes": node_count,
            "fixed_node_ratio": float(fixed_mask.mean()),
        }

        rta_values = frame["RTA"].to_numpy(dtype=np.float64, copy=False)
        rmises_values = frame["RMises"].to_numpy(dtype=np.float64, copy=False)
        pair_stats_raw.update(rta_values, rmises_values)
        pair_stats_log.update(np.log1p(rta_values), np.log1p(rmises_values))

        for target in TARGET_COLUMNS:
            values = frame[target].to_numpy(dtype=np.float64, copy=False)
            free_values = frame.loc[free_mask, target].to_numpy(dtype=np.float64, copy=False)
            fixed_values = frame.loc[fixed_mask, target].to_numpy(dtype=np.float64, copy=False)

            global_stats[target].update(values)
            counts[target]["global"] += int(values.size)

            if split_name in SPLIT_NAMES:
                split_stats[split_name][target].update(values)
                counts[target][split_name] += int(values.size)

            bc_group_stats["free_nodes"][target].update(free_values)
            bc_group_stats["fixed_nodes"][target].update(fixed_values)
            counts[target]["free_nodes"] += int(free_values.size)
            counts[target]["fixed_nodes"] += int(fixed_values.size)

            case_quantiles = compute_case_quantiles(values)
            row[f"{target}_mean"] = float(values.mean())
            row[f"{target}_std"] = float(values.std())
            row[f"{target}_max"] = float(values.max())
            for quantile_name, quantile_value in case_quantiles.items():
                row[f"{target}_{quantile_name}"] = quantile_value

        case_rows.append(row)

        if args.progress_every > 0 and (case_index % args.progress_every == 0 or case_index == total_cases):
            print(f"[pass1 {case_index}/{total_cases}] processed {case_name}")

    scope_paths = build_scope_paths(temp_dir)
    memmaps: dict[str, dict[str, np.memmap]] = {}
    offsets: dict[str, dict[str, int]] = {}
    for target in TARGET_COLUMNS:
        memmaps[target] = {}
        offsets[target] = {}
        for scope_name, count in counts[target].items():
            path = scope_paths[target][scope_name]
            if path.exists():
                path.unlink()
            memmaps[target][scope_name] = np.memmap(path, dtype=np.float64, mode="w+", shape=(count,))
            offsets[target][scope_name] = 0

    for case_index, case_name in enumerate(selected_case_names, start=1):
        frame = read_node_targets(available_cases[case_name])
        split_name = case_to_split.get(case_name, "unspecified")
        free_mask = frame["bc_mask"] == 0
        fixed_mask = ~free_mask

        for target in TARGET_COLUMNS:
            values = frame[target].to_numpy(dtype=np.float64, copy=False)
            size = int(values.size)
            start = offsets[target]["global"]
            memmaps[target]["global"][start:start + size] = values
            offsets[target]["global"] += size

            if split_name in SPLIT_NAMES:
                split_values = values
                start = offsets[target][split_name]
                memmaps[target][split_name][start:start + size] = split_values
                offsets[target][split_name] += size

            free_values = frame.loc[free_mask, target].to_numpy(dtype=np.float64, copy=False)
            fixed_values = frame.loc[fixed_mask, target].to_numpy(dtype=np.float64, copy=False)

            start = offsets[target]["free_nodes"]
            memmaps[target]["free_nodes"][start:start + int(free_values.size)] = free_values
            offsets[target]["free_nodes"] += int(free_values.size)

            start = offsets[target]["fixed_nodes"]
            memmaps[target]["fixed_nodes"][start:start + int(fixed_values.size)] = fixed_values
            offsets[target]["fixed_nodes"] += int(fixed_values.size)

        if args.progress_every > 0 and (case_index % args.progress_every == 0 or case_index == total_cases):
            print(f"[pass2 {case_index}/{total_cases}] processed {case_name}")

    exact_quantiles: dict[str, dict[str, dict[str, float]]] = {
        "global": {},
        **{split_name: {} for split_name in SPLIT_NAMES},
        **{group_name: {} for group_name in BC_GROUPS},
    }
    for target in TARGET_COLUMNS:
        for scope_name, mmap in memmaps[target].items():
            print(f"[sort] {target} {scope_name} ({mmap.shape[0]} values)")
            mmap.flush()
            mmap.sort()
            mmap.flush()
            exact_quantiles[scope_name][target] = exact_quantiles_from_sorted(mmap)

    joint_thresholds = {
        "RTA": exact_quantiles["global"]["RTA"]["q990"],
        "RMises": exact_quantiles["global"]["RMises"]["q990"],
    }
    hotspot_counts = {
        "count": 0,
        "rta_hot": 0,
        "rmises_hot": 0,
        "both_hot": 0,
    }

    for case_index, case_name in enumerate(selected_case_names, start=1):
        frame = read_node_targets(available_cases[case_name])
        rta_hot = frame["RTA"].to_numpy(dtype=np.float64, copy=False) >= joint_thresholds["RTA"]
        rmises_hot = frame["RMises"].to_numpy(dtype=np.float64, copy=False) >= joint_thresholds["RMises"]
        both_hot = rta_hot & rmises_hot
        hotspot_counts["count"] += int(len(frame))
        hotspot_counts["rta_hot"] += int(rta_hot.sum())
        hotspot_counts["rmises_hot"] += int(rmises_hot.sum())
        hotspot_counts["both_hot"] += int(both_hot.sum())
        if args.progress_every > 0 and (case_index % args.progress_every == 0 or case_index == total_cases):
            print(f"[pass3 {case_index}/{total_cases}] processed {case_name}")

    summary = {
        "dataset": {
            "root": str(dataset_root),
            "config_path": str(Path(args.config)),
            "output_dir": str(output_dir),
            "temp_dir": str(temp_dir),
            "num_cases": total_cases,
            "total_nodes": total_nodes,
            "split_sizes": {split_name: len(splits[split_name]) for split_name in SPLIT_NAMES},
        },
        "quantile_method": "exact full-data quantiles from sorted float64 memmaps",
        "global": {
            target: global_stats[target].to_summary(exact_quantiles["global"][target])
            for target in TARGET_COLUMNS
        },
        "splits": {
            split_name: {
                target: split_stats[split_name][target].to_summary(exact_quantiles[split_name][target])
                for target in TARGET_COLUMNS
            }
            for split_name in SPLIT_NAMES
        },
        "bc_groups": {
            group_name: {
                target: bc_group_stats[group_name][target].to_summary(exact_quantiles[group_name][target])
                for target in TARGET_COLUMNS
            }
            for group_name in BC_GROUPS
        },
        "joint": {
            "method": "exact Pearson correlation and exact hotspot overlap using full-data q99 thresholds",
            "pearson_raw": pair_stats_raw.pearson(),
            "pearson_log1p": pair_stats_log.pearson(),
            "rta_q99": joint_thresholds["RTA"],
            "rmises_q99": joint_thresholds["RMises"],
            "rta_hot_ratio": hotspot_counts["rta_hot"] / hotspot_counts["count"],
            "rmises_hot_ratio": hotspot_counts["rmises_hot"] / hotspot_counts["count"],
            "both_hot_ratio": hotspot_counts["both_hot"] / hotspot_counts["count"],
            "rmises_hot_given_rta_hot": hotspot_counts["both_hot"] / max(hotspot_counts["rta_hot"], 1),
            "rta_hot_given_rmises_hot": hotspot_counts["both_hot"] / max(hotspot_counts["rmises_hot"], 1),
        },
    }

    case_df = pd.DataFrame(case_rows).sort_values("case_name").reset_index(drop=True)
    case_df.to_csv(output_dir / "case_level_node_target_summary_exact.csv", index=False, encoding="utf-8")

    summary["case_level"] = {}
    for target in TARGET_COLUMNS:
        global_q99 = summary["global"][target]["quantiles"]["q990"]
        summary["case_level"][target] = {
            "top_by_p99": case_df.sort_values(f"{target}_p99", ascending=False).head(10).to_dict(orient="records"),
            "top_by_max": case_df.sort_values(f"{target}_max", ascending=False).head(10).to_dict(orient="records"),
            "aggregate": {
                "mean_case_mean": float(case_df[f"{target}_mean"].mean()),
                "median_case_mean": float(case_df[f"{target}_mean"].median()),
                "mean_case_p99": float(case_df[f"{target}_p99"].mean()),
                "median_case_p99": float(case_df[f"{target}_p99"].median()),
                "cases_case_p99_above_global_q99": int((case_df[f"{target}_p99"] > global_q99).sum()),
            },
        }

    write_json(output_dir / "node_target_distribution_summary_exact.json", summary)
    write_markdown_report(output_dir / "node_target_distribution_report_exact_zh.md", summary)

    for target in TARGET_COLUMNS:
        for mmap in memmaps[target].values():
            mmap.flush()

    print(f"Summary written to: {output_dir / 'node_target_distribution_summary_exact.json'}")
    print(f"Case-level CSV written to: {output_dir / 'case_level_node_target_summary_exact.csv'}")
    print(f"Report written to: {output_dir / 'node_target_distribution_report_exact_zh.md'}")


if __name__ == "__main__":
    main()

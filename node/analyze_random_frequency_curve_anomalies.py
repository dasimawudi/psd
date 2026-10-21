from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("node/outputs/random_frequency_curves_test_200each")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify and summarize abrupt downward dips in sampled frequency curves."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-column", type=str, default="target_raw")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help="Curve groups to analyze. Use 'all' to include every group.",
    )
    parser.add_argument(
        "--dip-window",
        type=int,
        default=2,
        help="Neighboring frequency samples per side used for the local baseline.",
    )
    parser.add_argument(
        "--dip-depth-log",
        type=float,
        default=3.0,
        help="A point is a deep dip when log1p(local baseline) - log1p(point) is at least this value.",
    )
    parser.add_argument(
        "--dip-neighbor-min",
        type=float,
        default=1.0,
        help="Ignore local dips whose two-sided baseline is below this raw response.",
    )
    parser.add_argument(
        "--floor-eps",
        type=float,
        default=1e-12,
        help="Values at or below this raw response are counted as floor/zero hits.",
    )
    parser.add_argument(
        "--max-floor-fraction",
        type=float,
        default=0.05,
        help="Threshold for reporting high floor/zero fractions.",
    )
    parser.add_argument(
        "--max-floor-run",
        type=int,
        default=3,
        help="Threshold for reporting long contiguous floor/zero runs.",
    )
    parser.add_argument(
        "--flag-floor-fraction",
        action="store_true",
        help="Also mark curves as anomalies when their floor/zero fraction exceeds --max-floor-fraction.",
    )
    parser.add_argument(
        "--flag-floor-run",
        action="store_true",
        help="Also mark curves as anomalies when their longest floor/zero run exceeds --max-floor-run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of most severe curves written into anomaly_summary.json.",
    )
    return parser.parse_args()


def _floor_runs(values: np.ndarray, frequencies: np.ndarray, floor_eps: float) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    is_floor = values <= float(floor_eps)
    start: int | None = None
    for i, flag in enumerate(is_floor):
        if bool(flag) and start is None:
            start = i
        if start is not None and (not bool(flag) or i == len(is_floor) - 1):
            end = i - 1 if not bool(flag) else i
            runs.append(
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "length": int(end - start + 1),
                    "start_frequency_hz": float(frequencies[start]),
                    "end_frequency_hz": float(frequencies[end]),
                }
            )
            start = None
    return runs


def _analyze_curve(
    group: pd.DataFrame,
    *,
    target_column: str,
    dip_window: int,
    dip_depth_log: float,
    dip_neighbor_min: float,
    floor_eps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = group.sort_values("frequency_hz").reset_index(drop=True)
    values = np.clip(ordered[target_column].to_numpy(dtype=np.float64), 0.0, None)
    logs = np.log1p(values)
    frequencies = ordered["frequency_hz"].to_numpy(dtype=np.float64)
    n_points = int(values.size)
    window = max(1, int(dip_window))

    dip_rows: list[dict[str, Any]] = []
    max_local_dip_log = 0.0
    deepest_dip_frequency_hz = float("nan")
    isolated_floor_dips = 0

    for i in range(1, max(n_points - 1, 1)):
        left = values[max(0, i - window) : i]
        right = values[i + 1 : min(n_points, i + 1 + window)]
        if left.size == 0 or right.size == 0:
            continue

        left_median = float(np.median(left))
        right_median = float(np.median(right))
        baseline = min(left_median, right_median)
        dip_depth = float(np.log1p(baseline) - logs[i])
        if dip_depth > max_local_dip_log:
            max_local_dip_log = dip_depth
            deepest_dip_frequency_hz = float(frequencies[i])

        if baseline < float(dip_neighbor_min) or dip_depth < float(dip_depth_log):
            continue

        is_floor_dip = bool(values[i] <= float(floor_eps))
        if is_floor_dip:
            isolated_floor_dips += 1
        dip_rows.append(
            {
                "curve_id": str(ordered.loc[i, "curve_id"]),
                "group": str(ordered.loc[i, "group"]),
                "case_name": str(ordered.loc[i, "case_name"]),
                "node_index": int(ordered.loc[i, "node_index"]),
                "frequency_hz": float(frequencies[i]),
                "target_raw": float(values[i]),
                "left_median": left_median,
                "right_median": right_median,
                "local_baseline": baseline,
                "dip_depth_log": dip_depth,
                "is_floor_dip": is_floor_dip,
            }
        )

    runs = _floor_runs(values, frequencies, floor_eps=float(floor_eps))
    floor_hits = int(np.count_nonzero(values <= float(floor_eps)))
    max_floor_run = max((run["length"] for run in runs), default=0)
    metrics = {
        "curve_id": str(ordered.loc[0, "curve_id"]),
        "group": str(ordered.loc[0, "group"]),
        "case_name": str(ordered.loc[0, "case_name"]),
        "node_index": int(ordered.loc[0, "node_index"]),
        "points": n_points,
        "peak_target": (
            float(ordered["peak_target"].iloc[0])
            if "peak_target" in ordered.columns
            else (float(values.max()) if n_points else 0.0)
        ),
        "peak_frequency_hz": (
            float(ordered["peak_frequency_hz"].iloc[0])
            if "peak_frequency_hz" in ordered.columns
            else float(frequencies[int(np.argmax(values))])
        ),
        "min_target": float(values.min()) if n_points else float("nan"),
        "median_target": float(np.median(values)) if n_points else float("nan"),
        "floor_hits": floor_hits,
        "floor_fraction": float(floor_hits / max(n_points, 1)),
        "floor_runs": int(len(runs)),
        "max_floor_run": int(max_floor_run),
        "deep_dip_points": int(len(dip_rows)),
        "isolated_floor_dips": int(isolated_floor_dips),
        "max_local_dip_log": float(max_local_dip_log),
        "deepest_dip_frequency_hz": deepest_dip_frequency_hz,
    }
    for run in runs:
        run["curve_id"] = metrics["curve_id"]
        run["group"] = metrics["group"]
        run["case_name"] = metrics["case_name"]
        run["node_index"] = metrics["node_index"]
    return metrics, dip_rows, runs


def _reason_list(
    row: pd.Series,
    *,
    max_floor_fraction: float,
    max_floor_run: int,
    flag_floor_fraction: bool,
    flag_floor_run: bool,
) -> list[str]:
    reasons: list[str] = []
    if int(row["deep_dip_points"]) > 0:
        reasons.append("deep_dip")
    if flag_floor_fraction and float(row["floor_fraction"]) > float(max_floor_fraction):
        reasons.append("floor_fraction")
    if flag_floor_run and int(row["max_floor_run"]) > int(max_floor_run):
        reasons.append("floor_run")
    return reasons


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "anomaly_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = input_dir / "selected_curve_points.csv"
    summary_path = input_dir / "selected_curve_summary.csv"
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    points = pd.read_csv(points_path)
    if args.target_column not in points.columns:
        raise ValueError(f"{points_path} does not contain {args.target_column!r}")

    groups = {str(group) for group in args.groups}
    if "all" not in groups:
        points = points[points["group"].astype(str).isin(groups)].copy()
    if points.empty:
        raise RuntimeError("No curve points remain after applying --groups.")

    metric_rows: list[dict[str, Any]] = []
    dip_rows: list[dict[str, Any]] = []
    floor_run_rows: list[dict[str, Any]] = []
    for _curve_id, curve_df in points.groupby("curve_id", sort=False):
        metrics, dips, runs = _analyze_curve(
            curve_df,
            target_column=args.target_column,
            dip_window=int(args.dip_window),
            dip_depth_log=float(args.dip_depth_log),
            dip_neighbor_min=float(args.dip_neighbor_min),
            floor_eps=float(args.floor_eps),
        )
        metric_rows.append(metrics)
        dip_rows.extend(dips)
        floor_run_rows.extend(runs)

    metrics_df = pd.DataFrame(metric_rows)
    reasons = metrics_df.apply(
        _reason_list,
        axis=1,
        max_floor_fraction=float(args.max_floor_fraction),
        max_floor_run=int(args.max_floor_run),
        flag_floor_fraction=bool(args.flag_floor_fraction),
        flag_floor_run=bool(args.flag_floor_run),
    )
    metrics_df["anomaly_reasons"] = reasons.map(lambda items: ";".join(items))
    metrics_df["is_anomaly"] = metrics_df["anomaly_reasons"].astype(bool)

    anomaly_df = metrics_df[metrics_df["is_anomaly"]].copy()
    metrics_df.to_csv(output_dir / "curve_anomaly_metrics.csv", index=False)
    pd.DataFrame(dip_rows).to_csv(output_dir / "deep_dip_points.csv", index=False)
    pd.DataFrame(floor_run_rows).to_csv(output_dir / "floor_runs.csv", index=False)
    anomaly_df.to_csv(output_dir / "anomaly_curves.csv", index=False)

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        merged = summary_df.merge(
            anomaly_df[
                [
                    "curve_id",
                    "anomaly_reasons",
                    "floor_fraction",
                    "max_floor_run",
                    "deep_dip_points",
                    "isolated_floor_dips",
                    "max_local_dip_log",
                    "deepest_dip_frequency_hz",
                ]
            ],
            on="curve_id",
            how="inner",
        )
        merged.to_csv(output_dir / "anomaly_curve_summary.csv", index=False)

    reason_counter: Counter[str] = Counter()
    for item in anomaly_df["anomaly_reasons"]:
        reason_counter.update(str(item).split(";"))

    top_columns = [
        "curve_id",
        "group",
        "case_name",
        "node_index",
        "anomaly_reasons",
        "max_local_dip_log",
        "deepest_dip_frequency_hz",
        "deep_dip_points",
        "isolated_floor_dips",
        "floor_fraction",
        "max_floor_run",
        "peak_target",
    ]
    top_curves = (
        anomaly_df.sort_values(
            ["deep_dip_points", "max_local_dip_log", "floor_fraction", "max_floor_run"],
            ascending=[False, False, False, False],
        )
        .head(int(args.top_k))[top_columns]
        .to_dict(orient="records")
    )

    report = {
        "input_dir": str(input_dir),
        "points_csv": str(points_path),
        "output_dir": str(output_dir),
        "target_column": args.target_column,
        "groups": sorted(groups),
        "thresholds": {
            "dip_window": int(args.dip_window),
            "dip_depth_log": float(args.dip_depth_log),
            "dip_neighbor_min": float(args.dip_neighbor_min),
            "floor_eps": float(args.floor_eps),
            "max_floor_fraction": float(args.max_floor_fraction),
            "max_floor_run": int(args.max_floor_run),
            "flag_floor_fraction": bool(args.flag_floor_fraction),
            "flag_floor_run": bool(args.flag_floor_run),
        },
        "total_curves": int(metrics_df.shape[0]),
        "anomaly_curves": int(anomaly_df.shape[0]),
        "normal_curves": int(metrics_df.shape[0] - anomaly_df.shape[0]),
        "anomaly_by_group": anomaly_df.groupby("group").size().to_dict(),
        "total_by_group": metrics_df.groupby("group").size().to_dict(),
        "anomaly_by_reason": dict(reason_counter),
        "floor_fraction_over_threshold": int(
            (metrics_df["floor_fraction"] > float(args.max_floor_fraction)).sum()
        ),
        "floor_run_over_threshold": int((metrics_df["max_floor_run"] > int(args.max_floor_run)).sum()),
        "anomaly_by_case_top20": anomaly_df.groupby("case_name").size().sort_values(ascending=False).head(20).to_dict(),
        "top_curves": top_curves,
        "outputs": {
            "curve_metrics_csv": str(output_dir / "curve_anomaly_metrics.csv"),
            "anomaly_curves_csv": str(output_dir / "anomaly_curves.csv"),
            "anomaly_curve_summary_csv": str(output_dir / "anomaly_curve_summary.csv"),
            "deep_dip_points_csv": str(output_dir / "deep_dip_points.csv"),
            "floor_runs_csv": str(output_dir / "floor_runs.csv"),
        },
    }
    _write_json(output_dir / "anomaly_summary.json", report)

    print(f"Analyzed {metrics_df.shape[0]} curves from {points_path}")
    print(f"Flagged {anomaly_df.shape[0]} anomaly curves")
    print(f"By group: {report['anomaly_by_group']}")
    print(f"By reason: {report['anomaly_by_reason']}")
    print(f"Wrote report to {output_dir}")


if __name__ == "__main__":
    main()

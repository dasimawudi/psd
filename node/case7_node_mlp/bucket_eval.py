from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import json

import numpy as np
import pandas as pd


DEFAULT_FREQUENCY_BINS = [20.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0, float("inf")]
DEFAULT_PEAK_BINS = [0.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, float("inf")]
DEFAULT_MEAN_BINS = [0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, float("inf")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket diagnostics from node MLP evaluation CSVs.")
    parser.add_argument("--diagnostics", type=str, required=True, help="Path to diagnostics CSV.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for bucket CSV/JSON outputs.")
    return parser.parse_args()


def _format_edge(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    if abs(value) >= 10000.0:
        return f"{value:.0e}"
    return f"{value:g}"


def _labels(edges: list[float]) -> list[str]:
    return [f"[{_format_edge(left)}, {_format_edge(right)})" for left, right in zip(edges[:-1], edges[1:])]


def _add_bucket(df: pd.DataFrame, column: str, edges: list[float], output_column: str) -> None:
    df[output_column] = pd.cut(
        df[column].astype(float),
        bins=edges,
        labels=_labels(edges),
        right=False,
        include_lowest=True,
    ).astype(str)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    value_array = values.astype(float).to_numpy()
    weight_array = weights.astype(float).to_numpy()
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(value_array[valid], weights=weight_array[valid]))


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    points = group["points"].astype(float)
    return {
        "samples": int(len(group)),
        "points": float(points.sum()),
        "frequency_min": float(group["frequency_hz"].min()),
        "frequency_max": float(group["frequency_hz"].max()),
        "target_peak_median": float(group["target_peak"].median()),
        "target_peak_p90": float(group["target_peak"].quantile(0.90)),
        "target_mean_median": float(group["target_mean"].median()),
        "log_mae_mean": _weighted_average(group["log_mae"], points),
        "mae_mean": _weighted_average(group["mae"], points),
        "within25_ratio_mean": _weighted_average(group["within25_ratio"], points),
        "peak_relative_error_mean": float(group["peak_relative_error"].mean()),
        "peak_relative_error_median": float(group["peak_relative_error"].median()),
        "peak_relative_error_p90": float(group["peak_relative_error"].quantile(0.90)),
        "top1_mae_mean": float(group["top1_mae"].mean()),
        "top1_log_mae_mean": float(group["top1_log_mae"].mean()) if "top1_log_mae" in group else float("nan"),
        "top5_mae_mean": float(group["top5_mae"].mean()),
        "top5_log_mae_mean": float(group["top5_log_mae"].mean()) if "top5_log_mae" in group else float("nan"),
        "pred_target_ratio_mean": float(group["pred_target_ratio"].mean()),
        "pred_target_ratio_median": float(group["pred_target_ratio"].median()),
        "under_pred_ratio_mean": _weighted_average(group["under_pred_ratio"], points),
    }


def summarize_by(df: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, group in df.groupby(bucket_column, dropna=False, sort=False):
        row = {"bucket_type": bucket_column, "bucket": str(bucket)}
        row.update(_summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    output_dir = Path(args.output_dir) if args.output_dir else diagnostics_path.parent / "bucket_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(diagnostics_path)
    if df.empty:
        raise ValueError(f"Diagnostics CSV is empty: {diagnostics_path}")
    df["pred_target_ratio"] = df["pred_mean"].astype(float) / df["target_mean"].astype(float).replace(0.0, np.nan)
    _add_bucket(df, "frequency_hz", DEFAULT_FREQUENCY_BINS, "frequency_bucket")
    _add_bucket(df, "target_peak", DEFAULT_PEAK_BINS, "target_peak_bucket")
    _add_bucket(df, "target_mean", DEFAULT_MEAN_BINS, "target_mean_bucket")

    outputs = {
        "frequency": summarize_by(df, "frequency_bucket"),
        "target_peak": summarize_by(df, "target_peak_bucket"),
        "target_mean": summarize_by(df, "target_mean_bucket"),
    }
    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}_buckets.csv", index=False)

    summary = {
        "diagnostics": str(diagnostics_path),
        "samples": int(len(df)),
        "points": float(df["points"].astype(float).sum()),
        "outputs": {name: str(output_dir / f"{name}_buckets.csv") for name in outputs},
        "overall": _summarize_group(df),
    }
    (output_dir / "bucket_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

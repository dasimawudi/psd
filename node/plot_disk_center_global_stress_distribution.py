from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_stress(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-2 or abs_value >= 1e4:
        return f"{value:.2e}"
    if abs_value < 10:
        return f"{value:.2f}"
    if abs_value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _quantile_label(value: float) -> str:
    return f"p{value * 100:g}".replace(".0", "")


def _threshold_by_label(summary: dict) -> dict[str, float]:
    quantiles = np.asarray(summary["quantiles"], dtype=np.float64)
    thresholds = np.asarray(summary["thresholds"], dtype=np.float64)
    return {_quantile_label(q): float(t) for q, t in zip(quantiles, thresholds)}


def _with_band_stress_ranges(labels: list[str], summary: dict) -> list[str]:
    thresholds = _threshold_by_label(summary)
    tick_labels: list[str] = []
    for label in labels:
        if "-" not in label:
            tick_labels.append(label)
            continue
        lower_label, upper_label = label.split("-", maxsplit=1)
        lower = thresholds.get(lower_label)
        upper = thresholds.get(upper_label)
        if lower is None or upper is None:
            tick_labels.append(label)
            continue
        tick_labels.append(f"{label}\n{_format_stress(lower)}-{_format_stress(upper)}")
    return tick_labels


def _with_cumulative_stress_thresholds(labels: list[str], summary: dict) -> list[str]:
    thresholds = _threshold_by_label(summary)
    tick_labels: list[str] = []
    for label in labels:
        threshold = thresholds.get(label)
        if threshold is None:
            tick_labels.append(label)
        else:
            tick_labels.append(f"{label}\n>={_format_stress(threshold)}")
    return tick_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot global stress distribution diagnostics for disk-center node MLP.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="node/outputs/node_mlp_v6_disk_center_baseline/target_quantile_within25_all",
        help="Directory containing target_quantile_within25_summary.json and CSVs.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated plots.")
    return parser.parse_args()


def _plot_quantile_thresholds(summary: dict, output_path: Path) -> None:
    quantiles = np.asarray(summary["quantiles"], dtype=np.float64)
    thresholds = np.asarray(summary["thresholds"], dtype=np.float64)
    labels = [_quantile_label(q) for q in quantiles]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(labels, thresholds, marker="o", linewidth=2.0, color="#2F6F9F")
    ax.set_yscale("log")
    ax.set_xlabel("target quantile")
    ax.set_ylabel("MISES_psd_density threshold, log scale")
    ax.set_title("Disk-center target stress long-tail thresholds")
    ax.grid(True, axis="y", alpha=0.28)
    for label, value in zip(labels, thresholds):
        ax.annotate(
            f"{value:.2g}",
            xy=(label, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_band_distribution(band_df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    labels = band_df["label"].astype(str).tolist()
    tick_labels = _with_band_stress_ranges(labels, summary)
    points = band_df["points"].astype(float).to_numpy()
    within25 = band_df["within25_ratio"].astype(float).to_numpy() * 100.0
    pred_ratio = band_df["pred_target_ratio"].astype(float).to_numpy()

    x = np.arange(len(labels))
    fig, ax_count = plt.subplots(figsize=(14.8, 6.7))
    bars = ax_count.bar(x, points, color="#5E7F6E", alpha=0.82, label="point count")
    ax_count.set_yscale("log")
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(tick_labels)
    ax_count.set_xlabel("target quantile band and stress range")
    ax_count.set_ylabel("points in band, log scale")
    ax_count.set_title("Disk-center point count and accuracy by target quantile band")
    ax_count.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, points):
        text = f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value:.0f}"
        ax_count.annotate(text, (bar.get_x() + bar.get_width() / 2.0, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, rotation=90)

    ax_acc = ax_count.twinx()
    ax_acc.plot(x, within25, color="#C23B32", marker="o", linewidth=2.0, label="within25")
    ax_acc.plot(x, pred_ratio * 100.0, color="#7A5195", marker="s", linewidth=1.7, label="pred/target x100")
    ax_acc.axhline(25.0, color="#6B6B6B", linestyle=":", linewidth=1.0)
    ax_acc.set_ylim(0.0, max(160.0, float(np.nanmax(pred_ratio * 100.0)) * 1.08))
    ax_acc.set_ylabel("within25 (%) / pred-target ratio (%)")
    for idx, value in zip(x, within25):
        ax_acc.annotate(
            f"{value:.1f}%",
            xy=(idx, value),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#C23B32",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    for idx, value in zip(x, pred_ratio * 100.0):
        ax_acc.annotate(
            f"{value:.0f}%",
            xy=(idx, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#7A5195",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    h1, l1 = ax_count.get_legend_handles_labels()
    h2, l2 = ax_acc.get_legend_handles_labels()
    ax_count.legend(h1 + h2, l1 + l2, loc="upper right")
    ax_count.tick_params(axis="x", labelrotation=28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tail_cumulative(cumulative_df: pd.DataFrame, summary: dict, output_path: Path) -> None:
    labels = cumulative_df["label"].astype(str).tolist()
    tick_labels = _with_cumulative_stress_thresholds(labels, summary)
    points = cumulative_df["points"].astype(float).to_numpy()
    within25 = cumulative_df["within25_ratio"].astype(float).to_numpy() * 100.0
    pred_ratio = cumulative_df["pred_target_ratio"].astype(float).to_numpy()
    miss25 = cumulative_df["miss25_rate"].astype(float).to_numpy() * 100.0

    x = np.arange(len(labels))
    fig, ax_count = plt.subplots(figsize=(13.6, 6.7))
    ax_count.bar(x, points, color="#4E79A7", alpha=0.78, label="points target >= threshold")
    ax_count.set_yscale("log")
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(tick_labels)
    ax_count.set_xlabel("cumulative target quantile threshold and stress cutoff")
    ax_count.set_ylabel("points, log scale")
    ax_count.set_title("Disk-center high-stress tail count and model quality")
    ax_count.grid(True, axis="y", alpha=0.25)

    ax_metric = ax_count.twinx()
    ax_metric.plot(x, within25, color="#C23B32", marker="o", linewidth=2.0, label="within25")
    ax_metric.plot(x, miss25, color="#E59F3A", marker="^", linewidth=1.7, label="miss25")
    ax_metric.plot(x, pred_ratio * 100.0, color="#7A5195", marker="s", linewidth=1.7, label="pred/target x100")
    ax_metric.set_ylim(0.0, 110.0)
    ax_metric.set_ylabel("metric (%)")
    for idx, value in zip(x, within25):
        ax_metric.annotate(
            f"{value:.1f}%",
            xy=(idx, value),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#C23B32",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    for idx, value in zip(x, pred_ratio * 100.0):
        ax_metric.annotate(
            f"{value:.0f}%",
            xy=(idx, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#7A5195",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    h1, l1 = ax_count.get_legend_handles_labels()
    h2, l2 = ax_metric.get_legend_handles_labels()
    ax_count.legend(h1 + h2, l1 + l2, loc="upper right")
    ax_count.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "global_stress_distribution_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((input_dir / "target_quantile_within25_summary.json").read_text(encoding="utf-8"))
    band_df = pd.read_csv(input_dir / "target_quantile_band_within25.csv")
    cumulative_df = pd.read_csv(input_dir / "target_quantile_cumulative_within25.csv")

    outputs = {
        "quantile_thresholds": output_dir / "disk_center_target_quantile_thresholds.png",
        "band_distribution": output_dir / "disk_center_target_band_distribution.png",
        "tail_cumulative": output_dir / "disk_center_tail_cumulative_quality.png",
    }
    _plot_quantile_thresholds(summary, outputs["quantile_thresholds"])
    _plot_band_distribution(band_df, summary, outputs["band_distribution"])
    _plot_tail_cumulative(cumulative_df, summary, outputs["tail_cumulative"])

    manifest = {
        "input_dir": str(input_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

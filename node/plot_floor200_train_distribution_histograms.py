from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REGION_LABELS = {
    "fullpart": "Fullpart",
    "earpiece": "Earpiece",
    "disk": "Disk",
}

LOG_TICKS = [0, 2.4, 4.8, 7.2, 9.6, 12.0, 14.4, 16.8, 19.2, 21.6, 24.0]


def _raw_tick_label(log_value: float) -> str:
    raw = math.expm1(float(log_value))
    return f"{raw:.1e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train target log1p histograms for floor200 report.")
    parser.add_argument("--hist-csv", required=True, type=str)
    parser.add_argument("--summary-json", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--floor-value", default=200.0, type=float)
    return parser.parse_args()


def _plot_region(df: pd.DataFrame, summary: dict, region: str, output_path: Path, floor_value: float) -> None:
    rows = df[df["region"] == region].sort_values("bin_index")
    centers = 0.5 * (rows["left_log1p"].to_numpy() + rows["right_log1p"].to_numpy())
    widths = rows["right_log1p"].to_numpy() - rows["left_log1p"].to_numpy()
    counts = rows["count"].to_numpy()
    floor_log = math.log1p(float(floor_value))
    stats = summary["summary"][region]

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.bar(centers, counts, width=widths * 0.96, color="#4C78A8", alpha=0.82, linewidth=0)
    ax.axvline(floor_log, color="#D62728", linestyle="--", linewidth=1.8, label=f"floor=200, log1p={floor_log:.3f}")
    ax.set_yscale("log")
    ax.set_xticks(LOG_TICKS)
    ax.set_xticklabels([_raw_tick_label(tick) for tick in LOG_TICKS], rotation=30, ha="right")
    ax.set_xlabel("MISES_psd_density, scientific notation; bins are equal width in log1p space")
    ax.set_ylabel("Point count, log scale")
    ax.set_title(
        f"{REGION_LABELS[region]} train target distribution, 100 bins | "
        f"points={int(stats['points']):,}, mean={stats['mean']:,.0f}, mean log1p={stats['mean_log1p']:.4f}"
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_combined(df: pd.DataFrame, summary: dict, output_path: Path, floor_value: float) -> None:
    floor_log = math.log1p(float(floor_value))
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 12.2), sharex=True)
    colors = {"fullpart": "#4C78A8", "earpiece": "#59A14F", "disk": "#F58518"}
    for ax, region in zip(axes, ["fullpart", "earpiece", "disk"], strict=True):
        rows = df[df["region"] == region].sort_values("bin_index")
        centers = 0.5 * (rows["left_log1p"].to_numpy() + rows["right_log1p"].to_numpy())
        widths = rows["right_log1p"].to_numpy() - rows["left_log1p"].to_numpy()
        stats = summary["summary"][region]
        ax.bar(centers, rows["count"].to_numpy(), width=widths * 0.96, color=colors[region], alpha=0.82, linewidth=0)
        ax.axvline(floor_log, color="#D62728", linestyle="--", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_ylabel(REGION_LABELS[region])
        ax.set_xticks(LOG_TICKS)
        ax.set_xticklabels([_raw_tick_label(tick) for tick in LOG_TICKS], rotation=30, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.text(
            0.995,
            0.88,
            f"points={int(stats['points']):,} | mean={stats['mean']:,.0f} | mean log1p={stats['mean_log1p']:.4f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.text(
            floor_log,
            0.94,
            "floor=200",
            color="#D62728",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#D62728", "alpha": 0.82, "boxstyle": "round,pad=0.18"},
        )
    axes[-1].set_xlabel("MISES_psd_density, scientific notation; bins are equal width in log1p space")
    fig.suptitle("Train target distribution by region, 100 log1p bins; red dashed line = floor 200", fontsize=13)
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _format_raw_range(left: float, right: float) -> str:
    if right >= 1e6:
        return f"{left:.3g} - {right:.3g}"
    if right >= 1e3:
        return f"{left:,.0f} - {right:,.0f}"
    if right >= 10:
        return f"{left:.2f} - {right:.2f}"
    return f"{left:.4g} - {right:.4g}"


def _plot_bin_range_table(df: pd.DataFrame, output_path: Path) -> None:
    bins = df[df["region"] == "fullpart"].sort_values("bin_index").reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 16.5))
    for panel_idx, ax in enumerate(axes):
        start = panel_idx * 25
        panel = bins.iloc[start : start + 25]
        rows = [
            [
                str(int(row.bin_index)),
                f"{float(row.left_log1p):.2f}-{float(row.right_log1p):.2f}",
                _format_raw_range(float(row.left_raw), float(row.right_raw)),
            ]
            for row in panel.itertuples(index=False)
        ]
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["bin", "log1p", "raw stress approx"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.16, 0.28, 0.56],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.4)
        table.scale(1.0, 1.42)
        for (row_idx, _col_idx), cell in table.get_celld().items():
            cell.set_linewidth(0.35)
            if row_idx == 0:
                cell.set_facecolor("#D9EAF7")
                cell.set_text_props(weight="bold")
        ax.set_title(f"bins {start}-{start + 24}", fontsize=11, pad=10)
    fig.suptitle("Histogram x-axis bin ranges: log1p(MISES_psd_density) -> raw stress", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.hist_csv)
    summary = json.loads(Path(args.summary_json).read_text())

    for region in ["fullpart", "earpiece", "disk"]:
        _plot_region(
            df,
            summary,
            region,
            output_dir / f"train_{region}_log1p_hist_100bins_floor200.png",
            args.floor_value,
        )
    _plot_combined(df, summary, output_dir / "train_fullpart_earpiece_disk_log1p_hist_100bins_floor200.png", args.floor_value)
    _plot_bin_range_table(df, output_dir / "x_axis_bin_stress_ranges_100bins.png")


if __name__ == "__main__":
    main()

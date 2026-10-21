from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_METRIC = "earpiece_stress_log_mae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/val loss and one training metric from history.csv.")
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history.csv. If omitted, --run-dir/history.csv is used.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Training output directory that contains history.csv.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help=(
            "Metric name without train_/val_ prefix, for example earpiece_stress_log_mae, "
            "earpiece_stress_within25_ratio, earpiece_stress_peak_relative_error, selection_score."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    parser.add_argument("--show", action="store_true", help="Show interactive window after saving.")
    return parser.parse_args()


def _resolve_history_path(args: argparse.Namespace) -> Path:
    if args.history is not None:
        return Path(args.history)
    if args.run_dir is not None:
        return Path(args.run_dir) / "history.csv"
    raise ValueError("Provide either --history or --run-dir.")


def _resolve_output_path(args: argparse.Namespace, history_path: Path) -> Path:
    if args.output is not None:
        return Path(args.output)
    return history_path.parent / f"training_curves_{args.metric}.png"


def _to_numeric_frame(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        raise FileNotFoundError(f"history.csv does not exist: {history_path}")
    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError(f"history.csv is empty: {history_path}")
    if "epoch" not in df.columns:
        raise KeyError(f"{history_path} does not contain an epoch column.")
    for column in df.columns:
        if column in {"selection_metric"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _metric_columns(df: pd.DataFrame, metric: str) -> list[tuple[str, str]]:
    if metric == "selection_score":
        return [("selection_score", "selection score")] if "selection_score" in df.columns else []

    columns: list[tuple[str, str]] = []
    train_column = f"train_{metric}"
    val_column = f"val_{metric}"
    if train_column in df.columns:
        columns.append((train_column, f"train {metric}"))
    if val_column in df.columns:
        columns.append((val_column, f"val {metric}"))
    if metric in df.columns:
        columns.append((metric, metric))
    return columns


def _available_metrics(df: pd.DataFrame) -> list[str]:
    metrics = set()
    for column in df.columns:
        if column.startswith("train_"):
            metrics.add(column.removeprefix("train_"))
        elif column.startswith("val_"):
            metrics.add(column.removeprefix("val_"))
        elif column == "selection_score":
            metrics.add(column)
    return sorted(metrics)


def plot_curves(history_path: Path, output_path: Path, metric: str, title: str | None, dpi: int, show: bool) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=not show)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required. Use the ci2n/yolo env or install matplotlib.") from exc

    df = _to_numeric_frame(history_path)
    epochs = df["epoch"]
    metric_columns = _metric_columns(df, metric)
    if not metric_columns:
        available = ", ".join(_available_metrics(df))
        raise KeyError(f"Metric '{metric}' was not found in {history_path}. Available metrics: {available}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    fig.suptitle(title or history_path.parent.name)

    loss_axis = axes[0]
    if "train_loss" in df.columns:
        loss_axis.plot(epochs, df["train_loss"], label="train loss", linewidth=1.8)
    if "val_loss" in df.columns:
        loss_axis.plot(epochs, df["val_loss"], label="val loss", linewidth=1.8)
    if "train_weighted_loss" in df.columns:
        loss_axis.plot(epochs, df["train_weighted_loss"], label="train weighted loss", linewidth=1.2, alpha=0.75)
    if "val_weighted_loss" in df.columns:
        loss_axis.plot(epochs, df["val_weighted_loss"], label="val weighted loss", linewidth=1.2, alpha=0.75)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Loss")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    metric_axis = axes[1]
    for column, label in metric_columns:
        metric_axis.plot(epochs, df[column], label=label, linewidth=1.8)
    metric_axis.set_xlabel("epoch")
    metric_axis.set_ylabel(metric)
    metric_axis.set_title(metric)
    metric_axis.grid(True, alpha=0.25)
    metric_axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main() -> None:
    args = parse_args()
    history_path = _resolve_history_path(args)
    output_path = _resolve_output_path(args, history_path)
    plot_curves(
        history_path=history_path,
        output_path=output_path,
        metric=args.metric,
        title=args.title,
        dpi=int(args.dpi),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()

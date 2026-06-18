from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.runtime import ensure_dir, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, build_model, prepare_point_sample
from case7_node_mlp.trace_node_frequency_response import (
    _auto_select_node,
    _load_single_node_raw_sample,
    _mode_frequencies,
    _node_label,
    _resolve_node_position,
)


FIELDNAMES = [
    "case_name",
    "node_index",
    "node_label",
    "frequency_hz",
    "curve_1_scaled",
    "curve_2_scaled",
    "curve_3_scaled",
    "curve_1_log_contribution",
    "curve_2_log_contribution",
    "curve_3_log_contribution",
    "mixing_weight_1",
    "mixing_weight_2",
    "mixing_weight_3",
    "weighted_curve_1_scaled",
    "weighted_curve_2_scaled",
    "weighted_curve_3_scaled",
    "node_scale_scaled",
    "low_rank_scaled",
    "pointwise_scaled",
    "regression_scaled",
    "low_rank_log",
    "pred_log",
    "pred_raw",
    "target_log",
    "target_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the latent low-rank curve values from a trained curve-head model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="node/outputs/node_mlp_v6_disk_center_low_rank_curves_fast_eval_continue70_4m/best.pt",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--case-name", type=str, default=None, help="Defaults to the first case in the selected split.")
    parser.add_argument("--node-index", type=int, default=None, help="Node row index/node_index to trace.")
    parser.add_argument(
        "--auto-node",
        choices=["peak", "top1", "worst_relative", "worst_nonresonance", "background"],
        default="peak",
        help="Node selection strategy if --node-index is omitted.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _load_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, StandardScaler, StandardScaler, dict[str, Any]]:
    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"]), feature_schema=feature_schema).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, x_scaler, y_scaler, feature_schema


def _plot_curve_values(path: Path, rows: list[dict[str, Any]], *, title: str, value_suffix: str, ylabel: str) -> None:
    df = pd.DataFrame(rows).sort_values("frequency_hz")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, color in enumerate(["tab:blue", "tab:orange", "tab:green"], start=1):
        ax.plot(
            df["frequency_hz"],
            df[f"curve_{idx}_{value_suffix}"],
            label=f"curve {idx}",
            linewidth=1.8,
            color=color,
        )
    ax.set_xlabel("Frequency Hz")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_mixing_and_prediction(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    df = pd.DataFrame(rows).sort_values("frequency_hz")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for idx, color in enumerate(["tab:blue", "tab:orange", "tab:green"], start=1):
        axes[0].plot(
            df["frequency_hz"],
            df[f"mixing_weight_{idx}"],
            label=f"w{idx}",
            linewidth=1.6,
            color=color,
        )
    axes[0].set_ylabel("Mixing weight")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for idx, color in enumerate(["tab:blue", "tab:orange", "tab:green"], start=1):
        axes[1].plot(
            df["frequency_hz"],
            df[f"weighted_curve_{idx}_scaled"],
            label=f"w{idx} * curve {idx}",
            linewidth=1.6,
            color=color,
        )
    axes[1].plot(df["frequency_hz"], df["node_scale_scaled"], label="node_scale", color="tab:purple", linewidth=1.5)
    axes[1].set_ylabel("Scaled contribution")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(df["frequency_hz"], df["target_raw"].clip(lower=1e-12), label="target", linewidth=1.8)
    axes[2].plot(df["frequency_hz"], df["pred_raw"].clip(lower=1e-12), label="prediction", linewidth=1.6)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Frequency Hz")
    axes[2].set_ylabel("MISES PSD density")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    device = resolve_device(args.device)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "low_rank_curve_values")
    model, x_scaler, y_scaler, feature_schema = _load_model(checkpoint, config, device)
    if not getattr(model, "low_rank_curve_head", False):
        raise RuntimeError("The checkpoint model does not use low_rank_curve_head.")
    if int(getattr(model, "curve_rank", 0)) != 3:
        raise RuntimeError(f"This plotting script expects curve_rank=3, got {getattr(model, 'curve_rank', None)}.")

    case_index = discover_case_index(dataset_cfg["root"])
    if args.case_name is None:
        split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
        case_name = split_names[args.split][0]
    else:
        case_name = args.case_name
    if case_name not in case_index:
        raise KeyError(f"Unknown case '{case_name}'.")
    case_dir = case_index[case_name]
    sample_paths = expand_case_sample_paths([case_dir], dataset_cfg)
    mode_frequencies = _mode_frequencies(case_dir)
    node_index = (
        _resolve_node_position(case_dir, int(args.node_index))
        if args.node_index is not None
        else _auto_select_node(
            case_dir,
            sample_paths,
            mode_frequencies,
            args.auto_node,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
        )
    )
    node_label = _node_label(case_dir, node_index)

    y_std = y_scaler.std.reshape(-1)[0].item()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample_path in sample_paths:
            raw = _load_single_node_raw_sample(
                sample_path=sample_path,
                dataset_cfg=dataset_cfg,
                feature_cfg=feature_cfg,
                node_index=node_index,
            )
            prepared = prepare_point_sample(
                raw,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                feature_schema=feature_schema,
                target_cfg=target_cfg,
                loss_cfg=loss_cfg,
            )
            features = prepared.features.to(device)
            output = model(features)
            if not isinstance(output, dict):
                raise RuntimeError("Expected a dict output from the low-rank curve-head model.")
            curve_values = output["curve_values"].detach().cpu().reshape(-1)
            mixing_weights = output["mixing_weights"].detach().cpu().reshape(-1)
            weighted = curve_values * mixing_weights
            node_scale = output["node_scale"].detach().cpu().reshape(-1)[0]
            low_rank = output["low_rank"].detach().cpu().reshape(-1)[0]
            pointwise = output["pointwise"].detach().cpu().reshape(-1)[0]
            regression = output["regression"].detach().cpu().reshape(-1, 1)
            low_rank_log, _ = _decode_prediction(output["low_rank"].detach().cpu(), y_scaler)
            pred_log, pred_raw = _decode_prediction(regression, y_scaler)

            rows.append(
                {
                    "case_name": case_name,
                    "node_index": int(node_index),
                    "node_label": int(node_label),
                    "frequency_hz": float(prepared.frequency_hz),
                    "curve_1_scaled": float(curve_values[0].item()),
                    "curve_2_scaled": float(curve_values[1].item()),
                    "curve_3_scaled": float(curve_values[2].item()),
                    "curve_1_log_contribution": float(curve_values[0].item() * y_std),
                    "curve_2_log_contribution": float(curve_values[1].item() * y_std),
                    "curve_3_log_contribution": float(curve_values[2].item() * y_std),
                    "mixing_weight_1": float(mixing_weights[0].item()),
                    "mixing_weight_2": float(mixing_weights[1].item()),
                    "mixing_weight_3": float(mixing_weights[2].item()),
                    "weighted_curve_1_scaled": float(weighted[0].item()),
                    "weighted_curve_2_scaled": float(weighted[1].item()),
                    "weighted_curve_3_scaled": float(weighted[2].item()),
                    "node_scale_scaled": float(node_scale.item()),
                    "low_rank_scaled": float(low_rank.item()),
                    "pointwise_scaled": float(pointwise.item()),
                    "regression_scaled": float(regression.reshape(-1)[0].item()),
                    "low_rank_log": float(low_rank_log.reshape(-1)[0].item()),
                    "pred_log": float(pred_log.reshape(-1)[0].item()),
                    "pred_raw": float(pred_raw.reshape(-1)[0].item()),
                    "target_log": float(prepared.target_log.reshape(-1)[0].item()),
                    "target_raw": float(prepared.target_raw.reshape(-1)[0].item()),
                }
            )

    rows.sort(key=lambda item: float(item["frequency_hz"]))
    if not rows:
        raise RuntimeError(f"No rows were traced for case={case_name}, node={node_index}.")

    stem = f"{case_name}_node{node_index}_{args.auto_node if args.node_index is None else 'specified'}"
    csv_path = output_dir / f"{stem}_low_rank_curve_values.csv"
    scaled_plot_path = output_dir / f"{stem}_curve_values_scaled.png"
    log_plot_path = output_dir / f"{stem}_curve_values_log_contribution.png"
    decomposition_plot_path = output_dir / f"{stem}_low_rank_decomposition.png"

    _write_csv(csv_path, rows)
    title = f"{case_name} node {node_index} low-rank curve values"
    _plot_curve_values(
        scaled_plot_path,
        rows,
        title=f"{title} | scaled target space",
        value_suffix="scaled",
        ylabel="Curve value in scaled log-target space",
    )
    _plot_curve_values(
        log_plot_path,
        rows,
        title=f"{title} | log contribution scale",
        value_suffix="log_contribution",
        ylabel="Curve contribution in log-target units",
    )
    _plot_mixing_and_prediction(decomposition_plot_path, rows, title=f"{title} | node mixture and prediction")

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "case_name": case_name,
        "node_index": int(node_index),
        "node_label": int(node_label),
        "samples": int(len(rows)),
        "y_scaler_mean": float(y_scaler.mean.reshape(-1)[0].item()),
        "y_scaler_std": float(y_std),
        "csv": str(csv_path),
        "scaled_curve_plot": str(scaled_plot_path),
        "log_contribution_curve_plot": str(log_plot_path),
        "decomposition_plot": str(decomposition_plot_path),
    }
    write_json(output_dir / f"{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

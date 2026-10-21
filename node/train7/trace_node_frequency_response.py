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
import pandas as pd
import torch

from train7.data import (
    PER_FREQUENCY_TARGET_COLUMN,
    RawPointSample,
    _build_base_features,
    _frequency_from_path,
    _load_aligned_target_column,
    _load_case_static,
    discover_case_index,
    expand_case_sample_paths,
    load_raw_point_sample,
    resolve_case_splits,
)
from train7.evaluate import _load_checkpoint
from train7.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from train7.scalers import StandardScaler
from train7.trainer import _decode_prediction, _effective_prediction_scaled, build_model, prepare_point_sample


TRACE_FIELDNAMES = [
    "case_name",
    "node_index",
    "node_label",
    "frequency_hz",
    "target_raw",
    "pred_raw",
    "target_log",
    "pred_log",
    "abs_error",
    "relative_error",
    "symmetric_relative_error",
    "log_abs_error",
    "within25_hit",
    "nearest_mode_frequency_hz",
    "nearest_mode_delta_hz",
    "nearest_mode_relative_delta",
    "is_near_mode_2pct",
    "is_near_mode_5pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one node's predicted frequency response for a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--case-name", type=str, default=None, help="Case directory name. Defaults to first case in split.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--node-index", type=int, default=None, help="Node row index/node_index to trace.")
    parser.add_argument(
        "--auto-node",
        choices=["peak", "top1", "worst_relative", "worst_nonresonance", "background"],
        default="peak",
        help="Node selection strategy if --node-index is omitted.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--point-batch-size", type=int, default=1048576)
    parser.add_argument(
        "--respect-selection",
        action="store_true",
        help="Trace only frequencies where the node survives dataset selection. By default, fixed-node tracing forces the node at every frequency.",
    )
    return parser.parse_args()


def _load_model(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device):
    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, x_scaler, y_scaler, feature_schema


def _mode_frequencies(case_dir: Path) -> np.ndarray:
    path = case_dir / "modal_frequencies.csv"
    if not path.exists():
        return np.array([], dtype=np.float32)
    df = pd.read_csv(path)
    if "frequency_hz" in df.columns:
        return df["frequency_hz"].to_numpy(dtype=np.float32)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.array([], dtype=np.float32)
    return numeric.iloc[:, -1].to_numpy(dtype=np.float32)


def _resolve_node_position(case_dir: Path, node_index: int) -> int:
    nodes = pd.read_csv(case_dir / "nodes.csv", usecols=lambda column: column in {"node_index"})
    if 0 <= int(node_index) < len(nodes):
        return int(node_index)
    if "node_index" in nodes.columns:
        row = nodes[nodes["node_index"].astype(int) == int(node_index)]
        if not row.empty:
            return int(row.index[0])
    raise IndexError(f"Node {node_index} is not a valid row index or node_index for {case_dir.name}.")


def _node_label(case_dir: Path, node_index: int) -> int:
    nodes = pd.read_csv(case_dir / "nodes.csv", usecols=lambda column: column in {"node_index", "node_label"})
    if 0 <= int(node_index) < len(nodes) and "node_label" in nodes.columns:
        return int(nodes.iloc[int(node_index)]["node_label"])
    if 0 <= int(node_index) < len(nodes) and "node_index" in nodes.columns:
        return int(nodes.iloc[int(node_index)]["node_index"])
    return int(node_index)


def _collect_case_targets(case_dir: Path, sample_paths: list[Path], dataset_cfg: dict[str, Any], feature_cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sample_paths:
        raw = load_raw_point_sample(path, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
        rows.append(
            pd.DataFrame(
                {
                    "frequency_hz": float(raw.frequency_hz),
                    "node_index": raw.node_indices.numpy().astype(np.int64, copy=False),
                    "target_raw": raw.target_raw.numpy().astype(np.float32, copy=False),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _auto_select_node(
    case_dir: Path,
    sample_paths: list[Path],
    mode_frequencies: np.ndarray,
    strategy: str,
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
) -> int:
    targets = _collect_case_targets(case_dir, sample_paths, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
    node_group = targets.groupby("node_index", sort=False)["target_raw"]
    if strategy == "peak":
        return int(targets.loc[targets["target_raw"].idxmax(), "node_index"])
    if strategy == "top1":
        max_by_node = node_group.max()
        threshold = float(max_by_node.quantile(0.99))
        candidates = max_by_node[max_by_node >= threshold]
        return int(candidates.sort_values(ascending=False).index[min(len(candidates) - 1, len(candidates) // 2)])
    if strategy == "background":
        stats = node_group.agg(["count", "mean", "std", "max"]).fillna({"std": 0.0})
        min_count = max(3, int(0.80 * len(sample_paths)))
        eligible = stats[(stats["count"] >= min_count) & (stats["max"] > 1e-12)]
        if eligible.empty:
            eligible = stats[stats["max"] > 1e-12]
        if eligible.empty:
            eligible = stats
        score = eligible["mean"].rank(pct=True) + eligible["std"].rank(pct=True)
        return int(score.sort_values().index[0])

    if mode_frequencies.size == 0:
        resonance_mask = np.zeros(len(targets), dtype=bool)
    else:
        frequencies = targets["frequency_hz"].to_numpy(dtype=np.float32)
        nearest_delta = np.min(np.abs(frequencies[:, None] - mode_frequencies[None, :]), axis=1)
        resonance_mask = nearest_delta / np.maximum(frequencies, 1e-6) <= 0.05
    if strategy == "worst_nonresonance":
        subset = targets.loc[~resonance_mask].copy()
        if subset.empty:
            subset = targets.copy()
    else:
        subset = targets.copy()
    node_max = subset.groupby("node_index", sort=False)["target_raw"].max()
    # This fallback selects a high-response node for later model-based tracing.
    return int(node_max.sort_values(ascending=False).index[0])


def _load_single_node_raw_sample(
    sample_path: Path,
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    node_index: int,
) -> RawPointSample:
    target_path = Path(sample_path)
    case_dir = target_path.parent.parent
    frequency_hz = _frequency_from_path(target_path)
    nodes_df, payload, _earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    payload["__case_dir__"] = str(case_dir)
    row_position = _resolve_node_position(case_dir, node_index)
    selected_indices = torch.tensor([row_position], dtype=torch.long)

    target_df = pd.read_csv(
        target_path,
        usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN},
    )
    target_values = _load_aligned_target_column(
        target_df,
        target_column=PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_df,
        target_path=target_path,
    )
    target_raw = torch.tensor(target_values[[row_position]], dtype=torch.float32)
    target_log = torch.log1p(target_raw.clamp_min(0.0)).unsqueeze(-1)
    geometry, scaled, masks, geometry_names, scaled_names, mask_names = _build_base_features(
        nodes_df=nodes_df,
        payload=payload,
        selected_indices=selected_indices,
        frequency_hz=frequency_hz,
        feature_cfg=feature_cfg,
    )
    return RawPointSample(
        name=f"{case_dir.name}/{target_path.name}",
        case_name=case_dir.name,
        frequency_hz=frequency_hz,
        geometry_features=geometry,
        scaled_features=scaled,
        mask_features=masks,
        target_log=target_log,
        target_raw=target_raw,
        node_indices=selected_indices,
        geometry_feature_names=geometry_names,
        scaled_feature_names=scaled_names,
        mask_feature_names=mask_names,
    )


def _predict_sample(
    *,
    model: Any,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    sample_path: Path,
    node_index: int,
    device: torch.device,
    respect_selection: bool,
) -> dict[str, float] | None:
    if respect_selection:
        raw = load_raw_point_sample(sample_path, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
    else:
        raw = _load_single_node_raw_sample(
            sample_path=sample_path,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            node_index=node_index,
        )
    matches = raw.node_indices == int(node_index)
    if not bool(matches.any()):
        return None
    prepared = prepare_point_sample(
        raw,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
    )
    row_index = int(torch.nonzero(matches, as_tuple=False)[0].item())
    feature = prepared.features[row_index : row_index + 1].to(device)
    with torch.no_grad():
        output = model(feature)
        prediction_scaled = _effective_prediction_scaled(
            output,
            y_scaler=y_scaler,
            sample_index=torch.zeros(1, dtype=torch.long, device=device),
            peak_relative_cfg=dict(getattr(model, "peak_relative_cfg", {})),
            zero_gate_cfg=dict(getattr(model, "zero_gate_cfg", {})),
        )
        pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled, y_scaler)
    target_raw = float(prepared.target_raw[row_index].item())
    pred_raw = float(pred_raw_t.item())
    target_log = float(prepared.target_log[row_index].item())
    pred_log = float(pred_log_t.item())
    abs_error = abs(pred_raw - target_raw)
    if abs(target_raw) > 1e-12:
        relative_error = abs_error / abs(target_raw)
        within25_hit = float(relative_error <= 0.25)
    else:
        relative_error = float("nan")
        within25_hit = float("nan")
    symmetric_relative_error = abs_error / max(0.5 * (abs(pred_raw) + abs(target_raw)), 1e-12)
    return {
        "frequency_hz": float(prepared.frequency_hz),
        "target_raw": target_raw,
        "pred_raw": pred_raw,
        "target_log": target_log,
        "pred_log": pred_log,
        "abs_error": abs_error,
        "relative_error": relative_error,
        "symmetric_relative_error": symmetric_relative_error,
        "log_abs_error": abs(pred_log - target_log),
        "within25_hit": within25_hit,
    }


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _plot_trace(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["frequency_hz"], df["target_raw"].clip(lower=1e-12), label="target", linewidth=1.8)
    axes[0].plot(df["frequency_hz"], df["pred_raw"].clip(lower=1e-12), label="prediction", linewidth=1.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MISES_psd_density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(df["frequency_hz"], df["relative_error"], label="relative error", linewidth=1.6)
    axes[1].axhline(0.25, color="tab:red", linestyle="--", linewidth=1.2, label="25%")
    near_mode = df[df["is_near_mode_5pct"].astype(bool)]
    if not near_mode.empty:
        axes[1].scatter(near_mode["frequency_hz"], near_mode["relative_error"], s=18, color="tab:orange", label="near mode <=5%")
    axes[1].set_xlabel("frequency Hz")
    axes[1].set_ylabel("relative error")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "node_frequency_traces")
    logger = make_logger(output_dir, logger_name="train7.trace_node_frequency_response", log_file="trace_node_frequency_response.log")
    device = resolve_device(args.device)
    model, x_scaler, y_scaler, feature_schema = _load_model(checkpoint, config, device)

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
    logger.info("Tracing node response | case=%s | node_index=%s | node_label=%s | samples=%s", case_name, node_index, node_label, len(sample_paths))

    rows: list[dict[str, Any]] = []
    for sample_path in sample_paths:
        row = _predict_sample(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            sample_path=sample_path,
            node_index=node_index,
            device=device,
            respect_selection=bool(args.respect_selection),
        )
        if row is None:
            continue
        frequency = row["frequency_hz"]
        if mode_frequencies.size:
            nearest_idx = int(np.argmin(np.abs(mode_frequencies - frequency)))
            nearest_frequency = float(mode_frequencies[nearest_idx])
            nearest_delta = abs(nearest_frequency - frequency)
        else:
            nearest_frequency = float("nan")
            nearest_delta = float("nan")
        row.update(
            {
                "case_name": case_name,
                "node_index": node_index,
                "node_label": node_label,
                "nearest_mode_frequency_hz": nearest_frequency,
                "nearest_mode_delta_hz": nearest_delta,
                "nearest_mode_relative_delta": nearest_delta / max(float(frequency), 1e-12),
                "is_near_mode_2pct": float(nearest_delta / max(float(frequency), 1e-12) <= 0.02),
                "is_near_mode_5pct": float(nearest_delta / max(float(frequency), 1e-12) <= 0.05),
            }
        )
        rows.append(row)

    rows.sort(key=lambda item: float(item["frequency_hz"]))
    if not rows:
        raise RuntimeError(f"Node {node_index} was not present in selected samples for {case_name}.")

    stem = f"{case_name}_node{node_index}_{args.auto_node if args.node_index is None else 'specified'}"
    csv_path = output_dir / f"{stem}_frequency_trace.csv"
    png_path = output_dir / f"{stem}_frequency_trace.png"
    _write_trace_csv(csv_path, rows)
    _plot_trace(png_path, rows, title=f"{case_name} node {node_index}")

    df = pd.DataFrame(rows)
    metric_mask = df["target_raw"].abs() > 1e-12
    near_mode_mask = df["is_near_mode_5pct"].astype(bool)
    metric_relative = df.loc[metric_mask, "relative_error"].astype(float)
    summary = {
        "case_name": case_name,
        "node_index": node_index,
        "node_label": node_label,
        "samples": int(len(df)),
        "relative_points": int(metric_mask.sum()),
        "zero_target_points": int((~metric_mask).sum()),
        "within25_ratio": float(df.loc[metric_mask, "within25_hit"].mean()) if bool(metric_mask.any()) else None,
        "relative_error_p50": float(metric_relative.quantile(0.50)) if bool(metric_mask.any()) else None,
        "relative_error_p90": float(metric_relative.quantile(0.90)) if bool(metric_mask.any()) else None,
        "near_mode_5pct_within25": float(df.loc[metric_mask & near_mode_mask, "within25_hit"].mean())
        if bool((metric_mask & near_mode_mask).any())
        else None,
        "non_near_mode_5pct_within25": float(df.loc[metric_mask & ~near_mode_mask, "within25_hit"].mean())
        if bool((metric_mask & ~near_mode_mask).any())
        else None,
        "target_peak_frequency_hz": float(df.loc[df["target_raw"].idxmax(), "frequency_hz"]),
        "pred_peak_frequency_hz": float(df.loc[df["pred_raw"].idxmax(), "frequency_hz"]),
        "target_peak": float(df["target_raw"].max()),
        "pred_peak": float(df["pred_raw"].max()),
        "csv": str(csv_path),
        "plot": str(png_path),
    }
    write_json(output_dir / f"{stem}_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

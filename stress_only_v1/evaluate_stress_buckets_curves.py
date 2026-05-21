from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from case7_gnn_stress_only.data import discover_case_index, expand_case_sample_paths
from case7_gnn_stress_only.runtime import ensure_dir, resolve_device, set_seed, write_json
from case7_gnn_stress_only.scalers import StandardScaler
from case7_gnn_stress_only.trainer import (
    build_model,
    compute_stress_hotspot_threshold,
    decode_field_prediction,
    get_stress_hotspot_metric_cfg,
    get_stress_peak_relative_cfg,
    get_two_stage_rmises_cfg,
)
from evaluate_stress_regions import (
    _case_dir_from_sample_path,
    _case_name_from_sample_path,
    _load_checkpoint,
    _load_global_payload,
    _load_prepared_case,
    _raw_node_features,
    _resolve_split_paths,
    _write_csv,
)


BUCKET_SAMPLE_FIELDNAMES = [
    "split",
    "sample",
    "case",
    "frequency_hz",
    "bucket",
    "node_count",
    "within25_count",
    "within25_ratio",
    "mae",
    "mean_relative_error",
    "target_mean",
    "pred_mean",
    "target_min",
    "target_max",
    "pred_max",
    "peak_relative_error",
]

BUCKET_SUMMARY_FIELDNAMES = [
    "split",
    "bucket",
    "node_count",
    "sample_count",
    "within25_count",
    "within25_ratio",
    "mae",
    "mean_relative_error",
    "target_mean",
    "pred_mean",
    "peak_relative_error",
]

CURVE_FIELDNAMES = [
    "case",
    "node_id",
    "node_group",
    "x",
    "y",
    "z",
    "frequency_hz",
    "target",
    "prediction",
    "absolute_error",
    "relative_error",
]


def _load_scalers(checkpoint: dict[str, Any]) -> dict[str, StandardScaler]:
    return {
        name: StandardScaler.from_state_dict(state)
        for name, state in checkpoint["scalers"].items()
    }


def _make_model(
    config: dict[str, Any],
    scalers: dict[str, StandardScaler],
    checkpoint: dict[str, Any],
    sample_path: Path,
    two_stage_cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    sample_case = _load_prepared_case(sample_path, config, scalers, two_stage_cfg)
    model = build_model(config, sample_case=sample_case).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _predict_sample(
    sample_path: Path,
    config: dict[str, Any],
    scalers: dict[str, StandardScaler],
    model: torch.nn.Module,
    device: torch.device,
    two_stage_cfg: dict[str, Any],
    stress_peak_relative_cfg: dict[str, Any],
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    batch = _load_prepared_case(sample_path, config, scalers, two_stage_cfg).to(device)
    with torch.no_grad():
        prediction = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            batch.global_features,
            node_graph_index=batch.node_graph_index,
            edge_graph_index=batch.edge_graph_index,
        )
        prediction_raw, _, _ = decode_field_prediction(
            prediction=prediction,
            target_scaler=scalers["target"].to(device),
            clamp_negative_rmises=bool(config["dataset"].get("clamp_negative_rmises", True)),
            two_stage_rmises_cfg=two_stage_cfg,
            stress_peak_relative_cfg=stress_peak_relative_cfg,
        )
    stress_pred = prediction_raw.detach().cpu()[:, 0].clamp_min(0.0)
    stress_target = batch.target_metric.detach().cpu()[:, 0].clamp_min(0.0)
    return batch, stress_target, stress_pred


def _bucket_mask(stress_target: torch.Tensor, bucket: str) -> torch.Tensor:
    if bucket == "all":
        return torch.ones_like(stress_target, dtype=torch.bool)
    if not bucket.startswith("top"):
        raise ValueError(f"Unsupported target bucket: {bucket}")
    fraction = float(bucket.removeprefix("top")) / 100.0
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"Invalid top bucket fraction: {bucket}")
    threshold = torch.quantile(stress_target, max(0.0, 1.0 - fraction))
    return stress_target >= threshold


def _bucket_metric_row(
    split: str,
    sample_name: str,
    case_name: str,
    frequency_hz: float | None,
    bucket: str,
    stress_target: torch.Tensor,
    stress_pred: torch.Tensor,
) -> dict[str, Any]:
    mask = _bucket_mask(stress_target, bucket)
    target = stress_target[mask]
    pred = stress_pred[mask]
    node_count = int(target.numel())
    if node_count <= 0:
        raise ValueError(f"Bucket {bucket} selected no nodes for sample {sample_name}")
    abs_error = (pred - target).abs()
    rel_error = abs_error / target.abs().clamp_min(1e-12)
    within_count = int((rel_error <= 0.25).sum().item())
    target_peak = float(target.max().item())
    pred_peak = float(pred.max().item())
    return {
        "split": split,
        "sample": sample_name,
        "case": case_name,
        "frequency_hz": frequency_hz,
        "bucket": bucket,
        "node_count": node_count,
        "within25_count": within_count,
        "within25_ratio": within_count / node_count,
        "mae": float(abs_error.mean().item()),
        "mean_relative_error": float(rel_error.mean().item()),
        "target_mean": float(target.mean().item()),
        "pred_mean": float(pred.mean().item()),
        "target_min": float(target.min().item()),
        "target_max": target_peak,
        "pred_max": pred_peak,
        "peak_relative_error": abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12),
    }


def _accumulate_bucket(summary: dict[tuple[str, str], dict[str, float]], row: dict[str, Any]) -> None:
    key = (str(row["split"]), str(row["bucket"]))
    stats = summary.setdefault(
        key,
        {
            "node_count": 0.0,
            "sample_count": 0.0,
            "within25_count": 0.0,
            "abs_error_sum": 0.0,
            "relative_error_sum": 0.0,
            "target_sum": 0.0,
            "pred_sum": 0.0,
            "peak_relative_error_sum": 0.0,
        },
    )
    node_count = float(row["node_count"])
    stats["node_count"] += node_count
    stats["sample_count"] += 1.0
    stats["within25_count"] += float(row["within25_count"])
    stats["abs_error_sum"] += float(row["mae"]) * node_count
    stats["relative_error_sum"] += float(row["mean_relative_error"]) * node_count
    stats["target_sum"] += float(row["target_mean"]) * node_count
    stats["pred_sum"] += float(row["pred_mean"]) * node_count
    stats["peak_relative_error_sum"] += float(row["peak_relative_error"])


def _finalize_bucket_summary(summary: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, bucket), stats in sorted(summary.items()):
        node_count = max(stats["node_count"], 1.0)
        sample_count = max(stats["sample_count"], 1.0)
        rows.append(
            {
                "split": split,
                "bucket": bucket,
                "node_count": int(stats["node_count"]),
                "sample_count": int(stats["sample_count"]),
                "within25_count": int(stats["within25_count"]),
                "within25_ratio": stats["within25_count"] / node_count,
                "mae": stats["abs_error_sum"] / node_count,
                "mean_relative_error": stats["relative_error_sum"] / node_count,
                "target_mean": stats["target_sum"] / node_count,
                "pred_mean": stats["pred_sum"] / node_count,
                "peak_relative_error": stats["peak_relative_error_sum"] / sample_count,
            }
        )
    return rows


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if not math.isfinite(value) or abs(in_max - in_min) <= 1e-12:
        return (out_min + out_max) / 2.0
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_curve_svg(
    path: Path,
    frequencies: list[float],
    target_values: list[float],
    pred_values: list[float],
    title: str,
    y_scale: str,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    if y_scale == "log1p":
        target_plot = [math.log1p(max(v, 0.0)) for v in target_values]
        pred_plot = [math.log1p(max(v, 0.0)) for v in pred_values]
        y_label = "log1p(MISES_psd_density)"
    elif y_scale == "linear":
        target_plot = target_values
        pred_plot = pred_values
        y_label = "MISES_psd_density"
    else:
        raise ValueError(f"Unsupported y scale: {y_scale}")

    x_min, x_max = min(frequencies), max(frequencies)
    y_min = min(target_plot + pred_plot)
    y_max = max(target_plot + pred_plot)
    if abs(y_max - y_min) <= 1e-12:
        y_max = y_min + 1.0

    target_points = [
        (
            _scale(freq, x_min, x_max, left, left + plot_w),
            _scale(value, y_min, y_max, top + plot_h, top),
        )
        for freq, value in zip(frequencies, target_plot)
    ]
    pred_points = [
        (
            _scale(freq, x_min, x_max, left, left + plot_w),
            _scale(value, y_min, y_max, top + plot_h, top),
        )
        for freq, value in zip(frequencies, pred_plot)
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="28" font-size="18" font-family="Arial" font-weight="700">{_svg_escape(title)}</text>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>
<text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-size="13" font-family="Arial">frequency (Hz)</text>
<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-size="13" font-family="Arial">{_svg_escape(y_label)}</text>
<text x="{left}" y="{top + plot_h + 20}" font-size="11" font-family="Arial">{x_min:.3g}</text>
<text x="{left + plot_w}" y="{top + plot_h + 20}" text-anchor="end" font-size="11" font-family="Arial">{x_max:.3g}</text>
<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-size="11" font-family="Arial">{y_min:.3g}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11" font-family="Arial">{y_max:.3g}</text>
<polyline points="{_polyline(target_points)}" fill="none" stroke="#d62728" stroke-width="2.2"/>
<polyline points="{_polyline(pred_points)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>
<rect x="{left + 15}" y="{top + 12}" width="185" height="45" fill="#fff" stroke="#ddd"/>
<line x1="{left + 28}" y1="{top + 28}" x2="{left + 62}" y2="{top + 28}" stroke="#d62728" stroke-width="2.5"/>
<text x="{left + 70}" y="{top + 32}" font-size="12" font-family="Arial">target</text>
<line x1="{left + 28}" y1="{top + 48}" x2="{left + 62}" y2="{top + 48}" stroke="#1f77b4" stroke-width="2.5"/>
<text x="{left + 70}" y="{top + 52}" font-size="12" font-family="Arial">prediction</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_hotspot_scatter_svg(
    path: Path,
    xy: torch.Tensor,
    hotspot_mask: torch.Tensor,
    selected: dict[str, list[int]],
    title: str,
    max_background_points: int = 8000,
) -> None:
    width, height = 760, 760
    margin = 55
    x = xy[:, 0].tolist()
    y = xy[:, 1].tolist()
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    indices = list(range(len(x)))
    if len(indices) > max_background_points:
        stride = max(1, math.ceil(len(indices) / max_background_points))
        indices = indices[::stride]
    selected_indices = sorted({idx for values in selected.values() for idx in values})
    indices = sorted(set(indices).union(selected_indices))

    def sx(value: float) -> float:
        return _scale(value, x_min, x_max, margin, width - margin)

    def sy(value: float) -> float:
        return _scale(value, y_min, y_max, height - margin, margin)

    circles = []
    for idx in indices:
        is_hot = bool(hotspot_mask[idx].item())
        fill = "#d62728" if is_hot else "#bdbdbd"
        opacity = "0.85" if is_hot else "0.35"
        radius = "2.2" if is_hot else "1.4"
        circles.append(
            f'<circle cx="{sx(x[idx]):.2f}" cy="{sy(y[idx]):.2f}" r="{radius}" fill="{fill}" opacity="{opacity}"/>'
        )
    colors = {"hotspot": "#111111", "non_hotspot": "#2ca02c"}
    for group, values in selected.items():
        color = colors.get(group, "#9467bd")
        for idx in values:
            circles.append(
                f'<circle cx="{sx(x[idx]):.2f}" cy="{sy(y[idx]):.2f}" r="5.0" fill="none" stroke="{color}" stroke-width="2.0"/>'
            )
            circles.append(
                f'<text x="{sx(x[idx]) + 6:.2f}" y="{sy(y[idx]) - 6:.2f}" font-size="10" font-family="Arial" fill="{color}">{idx}</text>'
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{margin}" y="30" font-size="18" font-family="Arial" font-weight="700">{_svg_escape(title)}</text>
<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="#fafafa" stroke="#ddd"/>
{''.join(circles)}
<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-size="12" font-family="Arial">x</text>
<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="12" font-family="Arial">y</text>
<circle cx="{margin + 18}" cy="{margin + 18}" r="4" fill="#d62728"/><text x="{margin + 30}" y="{margin + 22}" font-size="12" font-family="Arial">true hotspot</text>
<circle cx="{margin + 18}" cy="{margin + 38}" r="4" fill="#bdbdbd" opacity="0.6"/><text x="{margin + 30}" y="{margin + 42}" font-size="12" font-family="Arial">non hotspot</text>
<circle cx="{margin + 18}" cy="{margin + 58}" r="5" fill="none" stroke="#111" stroke-width="2"/><text x="{margin + 30}" y="{margin + 62}" font-size="12" font-family="Arial">selected hotspot node</text>
<circle cx="{margin + 18}" cy="{margin + 78}" r="5" fill="none" stroke="#2ca02c" stroke-width="2"/><text x="{margin + 30}" y="{margin + 82}" font-size="12" font-family="Arial">selected non-hotspot node</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _select_curve_case(
    config: dict[str, Any],
    split_paths: dict[str, list[Path]],
    case_name: str | None,
    case_dir: str | None,
    curve_split: str,
) -> Path:
    if case_dir:
        return Path(case_dir)
    case_index = discover_case_index(config["dataset"]["root"])
    if case_name:
        if case_name not in case_index:
            raise KeyError(f"Case {case_name!r} not found under {config['dataset']['root']}")
        return case_index[case_name]
    paths = split_paths[curve_split]
    if not paths:
        raise ValueError(f"No samples available for curve split: {curve_split}")
    return _case_dir_from_sample_path(paths[0])


def _case_frequency_paths(case_dir: Path, config: dict[str, Any]) -> list[Path]:
    paths = expand_case_sample_paths([case_dir], config["dataset"])
    return sorted(paths, key=lambda path: path.name)


def _node_coordinates(prepared_node_features: torch.Tensor, scalers: dict[str, StandardScaler], node_columns: list[str]) -> torch.Tensor:
    raw_features = _raw_node_features(prepared_node_features, scalers["node"])
    return raw_features[:, [node_columns.index("x"), node_columns.index("y"), node_columns.index("z")]]


def _select_curve_nodes(target_matrix: torch.Tensor, hotspot_count: int, non_hotspot_count: int) -> dict[str, list[int]]:
    peak_by_node = target_matrix.max(dim=0).values
    hotspot_count = max(0, min(int(hotspot_count), peak_by_node.numel()))
    non_hotspot_count = max(0, min(int(non_hotspot_count), peak_by_node.numel()))
    hotspot_nodes = torch.topk(peak_by_node, k=hotspot_count, largest=True).indices.tolist() if hotspot_count else []
    median_peak = torch.quantile(peak_by_node, 0.50)
    non_pool = torch.nonzero(peak_by_node <= median_peak, as_tuple=False).flatten()
    if non_pool.numel() == 0:
        non_pool = torch.arange(peak_by_node.numel())
    non_values = peak_by_node[non_pool]
    order = torch.argsort(non_values, descending=True)
    non_hotspot_nodes = non_pool[order[:non_hotspot_count]].tolist() if non_hotspot_count else []
    return {"hotspot": [int(v) for v in hotspot_nodes], "non_hotspot": [int(v) for v in non_hotspot_nodes]}


def _write_case_curves(
    output_dir: Path,
    config: dict[str, Any],
    scalers: dict[str, StandardScaler],
    model: torch.nn.Module,
    device: torch.device,
    two_stage_cfg: dict[str, Any],
    stress_peak_relative_cfg: dict[str, Any],
    hotspot_metric_cfg: dict[str, Any],
    case_dir: Path,
    hotspot_nodes: int,
    non_hotspot_nodes: int,
    y_scale: str,
) -> dict[str, Any]:
    curve_dir = ensure_dir(output_dir / "case_node_curves")
    case_paths = _case_frequency_paths(case_dir, config)
    frequencies: list[float] = []
    targets: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    first_batch = None

    for sample_path in case_paths:
        batch, stress_target, stress_pred = _predict_sample(
            sample_path=sample_path,
            config=config,
            scalers=scalers,
            model=model,
            device=device,
            two_stage_cfg=two_stage_cfg,
            stress_peak_relative_cfg=stress_peak_relative_cfg,
        )
        first_batch = first_batch or batch
        frequencies.append(float(batch.frequency_hz if batch.frequency_hz is not None else len(frequencies)))
        targets.append(stress_target)
        preds.append(stress_pred)

    if not targets:
        raise ValueError(f"No frequency samples found for case: {case_dir}")

    target_matrix = torch.stack(targets, dim=0)
    pred_matrix = torch.stack(preds, dim=0)
    selected = _select_curve_nodes(target_matrix, hotspot_nodes, non_hotspot_nodes)
    assert first_batch is not None
    coords = _node_coordinates(
        first_batch.node_features.detach().cpu(),
        scalers=scalers,
        node_columns=list(config["dataset"]["node_columns"]),
    )
    hotspot_threshold = compute_stress_hotspot_threshold(target_matrix.max(dim=0).values, hotspot_metric_cfg)
    hotspot_mask = target_matrix.max(dim=0).values >= hotspot_threshold

    curve_rows: list[dict[str, Any]] = []
    case_name = case_dir.name
    for group, node_ids in selected.items():
        for node_id in node_ids:
            node_target = target_matrix[:, node_id].tolist()
            node_pred = pred_matrix[:, node_id].tolist()
            x, y, z = coords[node_id].tolist()
            for freq, target, pred in zip(frequencies, node_target, node_pred):
                abs_error = abs(float(pred) - float(target))
                curve_rows.append(
                    {
                        "case": case_name,
                        "node_id": node_id,
                        "node_group": group,
                        "x": x,
                        "y": y,
                        "z": z,
                        "frequency_hz": freq,
                        "target": target,
                        "prediction": pred,
                        "absolute_error": abs_error,
                        "relative_error": abs_error / max(abs(float(target)), 1e-12),
                    }
                )
            _write_curve_svg(
                curve_dir / f"{case_name}_{group}_node{node_id}_{y_scale}.svg",
                frequencies=frequencies,
                target_values=[float(v) for v in node_target],
                pred_values=[float(v) for v in node_pred],
                title=f"{case_name} | {group} node {node_id}",
                y_scale=y_scale,
            )

    _write_csv(curve_dir / f"{case_name}_node_curve_values.csv", curve_rows, CURVE_FIELDNAMES)
    _write_hotspot_scatter_svg(
        curve_dir / f"{case_name}_hotspot_non_hotspot_xy.svg",
        xy=coords[:, :2],
        hotspot_mask=hotspot_mask,
        selected=selected,
        title=f"{case_name} hotspot/non-hotspot nodes",
    )
    return {
        "case": case_name,
        "frequency_count": len(frequencies),
        "selected_nodes": selected,
        "curve_values_csv": str(curve_dir / f"{case_name}_node_curve_values.csv"),
        "scatter_svg": str(curve_dir / f"{case_name}_hotspot_non_hotspot_xy.svg"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate target top buckets and plot node frequency curves.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--device", type=str, default=None, help="Override evaluation device.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples per split for bucket metrics.")
    parser.add_argument("--buckets", type=str, default="all,top1,top5,top10")
    parser.add_argument("--case-name", type=str, default=None, help="Case folder name for curve plots.")
    parser.add_argument("--case-dir", type=str, default=None, help="Explicit case directory for curve plots.")
    parser.add_argument("--curve-split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--hotspot-nodes", type=int, default=3)
    parser.add_argument("--non-hotspot-nodes", type=int, default=3)
    parser.add_argument("--curve-y-scale", type=str, default="log1p", choices=("log1p", "linear"))
    parser.add_argument("--skip-curves", action="store_true", help="Only compute bucket metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    config["training"] = dict(config["training"])
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "bucket_curve_eval")
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device or config["training"]["device"])
    scalers = _load_scalers(checkpoint)
    two_stage_cfg = get_two_stage_rmises_cfg(config)
    stress_peak_relative_cfg = get_stress_peak_relative_cfg(config)
    hotspot_metric_cfg = get_stress_hotspot_metric_cfg(config)
    split_paths = _resolve_split_paths(config)
    splits = list(split_paths) if args.split == "all" else [args.split]
    first_split = next(split_name for split_name in splits if split_paths[split_name])
    first_sample_path = split_paths[first_split][0]
    model = _make_model(config, scalers, checkpoint, first_sample_path, two_stage_cfg, device)

    bucket_names = [part.strip() for part in args.buckets.split(",") if part.strip()]
    sample_rows: list[dict[str, Any]] = []
    summary_accumulator: dict[tuple[str, str], dict[str, float]] = {}

    for split in splits:
        paths = split_paths[split]
        if args.max_samples is not None:
            paths = paths[: max(0, int(args.max_samples))]
        for sample_path in paths:
            batch, stress_target, stress_pred = _predict_sample(
                sample_path=sample_path,
                config=config,
                scalers=scalers,
                model=model,
                device=device,
                two_stage_cfg=two_stage_cfg,
                stress_peak_relative_cfg=stress_peak_relative_cfg,
            )
            case_name = _case_name_from_sample_path(sample_path)
            for bucket in bucket_names:
                row = _bucket_metric_row(
                    split=split,
                    sample_name=batch.name,
                    case_name=case_name,
                    frequency_hz=batch.frequency_hz,
                    bucket=bucket,
                    stress_target=stress_target,
                    stress_pred=stress_pred,
                )
                sample_rows.append(row)
                _accumulate_bucket(summary_accumulator, row)

    summary_rows = _finalize_bucket_summary(summary_accumulator)
    _write_csv(output_dir / "target_bucket_sample_metrics.csv", sample_rows, BUCKET_SAMPLE_FIELDNAMES)
    _write_csv(output_dir / "target_bucket_summary_metrics.csv", summary_rows, BUCKET_SUMMARY_FIELDNAMES)

    curve_summary: dict[str, Any] | None = None
    if not args.skip_curves:
        curve_case_dir = _select_curve_case(
            config=config,
            split_paths=split_paths,
            case_name=args.case_name,
            case_dir=args.case_dir,
            curve_split=args.curve_split,
        )
        curve_summary = _write_case_curves(
            output_dir=output_dir,
            config=config,
            scalers=scalers,
            model=model,
            device=device,
            two_stage_cfg=two_stage_cfg,
            stress_peak_relative_cfg=stress_peak_relative_cfg,
            hotspot_metric_cfg=hotspot_metric_cfg,
            case_dir=curve_case_dir,
            hotspot_nodes=args.hotspot_nodes,
            non_hotspot_nodes=args.non_hotspot_nodes,
            y_scale=args.curve_y_scale,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "buckets": bucket_names,
        "bucket_summary_csv": str(output_dir / "target_bucket_summary_metrics.csv"),
        "bucket_sample_csv": str(output_dir / "target_bucket_sample_metrics.csv"),
        "curve_summary": curve_summary,
    }
    write_json(output_dir / "bucket_curve_eval_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

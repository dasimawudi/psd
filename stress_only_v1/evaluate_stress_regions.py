from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from case7_gnn_stress_only.data import (
    discover_case_index,
    expand_case_sample_paths,
    load_case_graph,
    resolve_case_splits,
)
from case7_gnn_stress_only.runtime import ensure_dir, resolve_device, set_seed, write_json
from case7_gnn_stress_only.scalers import StandardScaler
from case7_gnn_stress_only.trainer import (
    PreparedCase,
    build_model,
    decode_field_prediction,
    get_mode_shape_loader_kwargs,
    get_stress_hotspot_metric_cfg,
    get_stress_peak_relative_cfg,
    get_two_stage_rmises_cfg,
    prepare_case,
)


REGION_FIELDNAMES = [
    "split",
    "sample",
    "case",
    "frequency_hz",
    "region",
    "node_count",
    "mae",
    "rmse",
    "within25_count",
    "within25_ratio",
    "miss25_rate",
    "mean_relative_error",
    "median_relative_error",
    "p90_relative_error",
    "p95_relative_error",
    "max_relative_error",
    "mean_symmetric_relative_error",
    "median_symmetric_relative_error",
    "max_abs_error",
    "under_pred_count",
    "under_pred_ratio",
    "over_pred_count",
    "over_pred_ratio",
    "target_mean",
    "pred_mean",
    "bias",
    "target_peak",
    "pred_peak",
    "peak_relative_error",
    "r2",
]


SUMMARY_FIELDNAMES = [
    "split",
    "region",
    "node_count",
    "sample_count",
    "mae",
    "rmse",
    "within25_count",
    "within25_ratio",
    "miss25_rate",
    "mean_relative_error",
    "mean_symmetric_relative_error",
    "mean_sample_median_relative_error",
    "mean_sample_p90_relative_error",
    "mean_sample_p95_relative_error",
    "under_pred_count",
    "under_pred_ratio",
    "over_pred_count",
    "over_pred_ratio",
    "target_mean",
    "pred_mean",
    "bias",
    "peak_relative_error",
    "r2",
]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _case_dir_from_sample_path(sample_path: Path) -> Path:
    if sample_path.is_dir():
        return sample_path
    if sample_path.parent.name == "per_frequency_mises":
        return sample_path.parent.parent
    raise ValueError(f"Unsupported sample path: {sample_path}")


def _case_name_from_sample_path(sample_path: Path) -> str:
    return _case_dir_from_sample_path(sample_path).name


def _load_global_payload(sample_path: Path) -> dict[str, Any]:
    with (_case_dir_from_sample_path(sample_path) / "global.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _param_value(payload: dict[str, Any], key: str, fallback_index: int | None = None, default: float = 0.0) -> float:
    params = payload.get("params", {})
    if key in params:
        return float(params[key])
    params_list = payload.get("params_list", [])
    if fallback_index is not None and fallback_index < len(params_list):
        return float(params_list[fallback_index])
    return float(default)


def _circle_centers(count: int, radius: float, dtype: torch.dtype) -> torch.Tensor:
    if count <= 0 or radius <= 0.0:
        return torch.empty((0, 2), dtype=dtype)
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=count + 1, dtype=dtype)[:-1]
    return torch.stack([-radius * torch.sin(angles), radius * torch.cos(angles)], dim=1)


def _nearest_center_distance(xy: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if centers.numel() == 0:
        return torch.full((xy.size(0),), float("inf"), dtype=xy.dtype)
    return torch.linalg.norm(xy[:, None, :] - centers[None, :, :], dim=-1).amin(dim=1)


def _raw_node_features(batch_node_features: torch.Tensor, node_scaler: StandardScaler) -> torch.Tensor:
    return node_scaler.inverse_transform(batch_node_features.detach().cpu())


def _region_masks(
    raw_node_features: torch.Tensor,
    node_columns: list[str],
    payload: dict[str, Any],
    ear_hole_factor: float,
    plate_hole_factor: float,
    root_band_mm: float | None,
) -> dict[str, torch.Tensor]:
    x_index = node_columns.index("x")
    y_index = node_columns.index("y")
    x = raw_node_features[:, x_index]
    y = raw_node_features[:, y_index]
    xy = torch.stack([x, y], dim=1)
    r = torch.linalg.norm(xy, dim=1)

    fixed_geometry = payload.get("fixed_geometry", {})
    plate_radius = _param_value(payload, "plate_radius", fallback_index=6, default=0.0)
    earpiece_radial_dist = _param_value(payload, "earpiece_RadialDist", fallback_index=1, default=0.0)
    earpiece_top_width = _param_value(payload, "earpiece_TopWidth", fallback_index=2, default=20.0)
    mass_couple_radius = float(fixed_geometry.get("mass_couple_radius", 65.0))
    earpiece_count = int(fixed_geometry.get("earpiece_Count_default", 3))
    earpiece_hole_radius = float(fixed_geometry.get("earpiece_HoleRadius", 4.0))
    plate_hole_count = int(fixed_geometry.get("plate_HoleCount", 0))
    plate_hole_radius = float(fixed_geometry.get("plate_HoleRadius", 0.0))
    plate_hole_dist = float(fixed_geometry.get("plate_HoleDist", 0.0))
    root_band = float(root_band_mm) if root_band_mm is not None else max(float(earpiece_top_width), 20.0)

    disk = r <= plate_radius
    earpiece = r > plate_radius
    center_couple = r <= mass_couple_radius
    disk_outer = disk & ~center_couple
    earpiece_root = earpiece & (r <= plate_radius + root_band)

    dtype = raw_node_features.dtype
    ear_centers = _circle_centers(earpiece_count, earpiece_radial_dist, dtype=dtype)
    dist_to_ear_hole = _nearest_center_distance(xy, ear_centers)
    ear_hole_neighborhood = dist_to_ear_hole <= max(earpiece_hole_radius * ear_hole_factor, 1e-6)
    earpiece_body = earpiece & ~ear_hole_neighborhood

    plate_centers = _circle_centers(plate_hole_count, plate_hole_dist, dtype=dtype)
    dist_to_plate_hole = _nearest_center_distance(xy, plate_centers)
    plate_hole_neighborhood = dist_to_plate_hole <= max(plate_hole_radius * plate_hole_factor, 1e-6)

    masks = {
        "all": torch.ones(raw_node_features.size(0), dtype=torch.bool),
        "disk": disk,
        "center_couple": center_couple,
        "disk_outer": disk_outer,
        "plate_hole_neighborhood": plate_hole_neighborhood,
        "earpiece": earpiece,
        "earpiece_root": earpiece_root,
        "earpiece_hole_neighborhood": ear_hole_neighborhood,
        "earpiece_body": earpiece_body,
    }
    if "bc_mask" in node_columns:
        bc_index = node_columns.index("bc_mask")
        masks["fixed_bc"] = raw_node_features[:, bc_index] >= 0.5
    return masks


def _metric_row(
    split: str,
    sample: str,
    case_name: str,
    frequency_hz: float | None,
    region: str,
    stress_target: torch.Tensor,
    stress_pred: torch.Tensor,
    mask: torch.Tensor,
    within_relative_error: float,
) -> dict[str, Any] | None:
    node_count = int(mask.sum().item())
    if node_count <= 0:
        return None

    target = stress_target[mask]
    pred = stress_pred[mask]
    signed_error = pred - target
    abs_error = signed_error.abs()
    squared_error = signed_error.pow(2.0)
    relative_error = abs_error / target.abs().clamp_min(1e-12)
    symmetric_relative_error = 2.0 * abs_error / (pred.abs() + target.abs()).clamp_min(1e-12)
    within_count = int((relative_error <= within_relative_error).sum().item())
    under_pred = pred < target
    over_pred = pred > target
    under_pred_count = int(under_pred.sum().item())
    over_pred_count = int(over_pred.sum().item())
    target_peak = float(target.max().item()) if target.numel() else 0.0
    pred_peak = float(pred.max().item()) if pred.numel() else 0.0
    peak_relative_error = abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12)
    target_centered = target - target.mean()
    sst = float(target_centered.pow(2.0).sum().item())
    sse = float(squared_error.sum().item())
    r2 = None if sst <= 1e-12 else 1.0 - sse / sst

    return {
        "split": split,
        "sample": sample,
        "case": case_name,
        "frequency_hz": frequency_hz,
        "region": region,
        "node_count": node_count,
        "mae": float(abs_error.mean().item()),
        "rmse": float(torch.sqrt(squared_error.mean()).item()),
        "within25_count": within_count,
        "within25_ratio": within_count / max(node_count, 1),
        "miss25_rate": 1.0 - within_count / max(node_count, 1),
        "mean_relative_error": float(relative_error.mean().item()),
        "median_relative_error": float(torch.quantile(relative_error, 0.50).item()),
        "p90_relative_error": float(torch.quantile(relative_error, 0.90).item()),
        "p95_relative_error": float(torch.quantile(relative_error, 0.95).item()),
        "max_relative_error": float(relative_error.max().item()),
        "mean_symmetric_relative_error": float(symmetric_relative_error.mean().item()),
        "median_symmetric_relative_error": float(torch.quantile(symmetric_relative_error, 0.50).item()),
        "max_abs_error": float(abs_error.max().item()),
        "under_pred_count": under_pred_count,
        "under_pred_ratio": under_pred_count / max(node_count, 1),
        "over_pred_count": over_pred_count,
        "over_pred_ratio": over_pred_count / max(node_count, 1),
        "target_mean": float(target.mean().item()),
        "target_sq_mean": float(target.pow(2.0).mean().item()),
        "pred_mean": float(pred.mean().item()),
        "bias": float(signed_error.mean().item()),
        "target_peak": target_peak,
        "pred_peak": pred_peak,
        "peak_relative_error": peak_relative_error,
        "r2": r2,
    }


def _accumulate(summary: dict[tuple[str, str], dict[str, float]], row: dict[str, Any]) -> None:
    key = (str(row["split"]), str(row["region"]))
    stats = summary.setdefault(
        key,
        {
            "node_count": 0.0,
            "abs_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "within25_count": 0.0,
            "relative_error_sum": 0.0,
            "symmetric_relative_error_sum": 0.0,
            "sample_median_relative_error_weighted_sum": 0.0,
            "sample_p90_relative_error_weighted_sum": 0.0,
            "sample_p95_relative_error_weighted_sum": 0.0,
            "under_pred_count": 0.0,
            "over_pred_count": 0.0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
            "pred_sum": 0.0,
            "bias_sum": 0.0,
            "peak_relative_error_sum": 0.0,
            "sample_count": 0.0,
        },
    )
    node_count = float(row["node_count"])
    stats["node_count"] += node_count
    stats["abs_error_sum"] += float(row["mae"]) * node_count
    stats["squared_error_sum"] += float(row["rmse"]) ** 2.0 * node_count
    stats["within25_count"] += float(row["within25_count"])
    stats["relative_error_sum"] += float(row["mean_relative_error"]) * node_count
    stats["symmetric_relative_error_sum"] += float(row["mean_symmetric_relative_error"]) * node_count
    stats["sample_median_relative_error_weighted_sum"] += float(row["median_relative_error"]) * node_count
    stats["sample_p90_relative_error_weighted_sum"] += float(row["p90_relative_error"]) * node_count
    stats["sample_p95_relative_error_weighted_sum"] += float(row["p95_relative_error"]) * node_count
    stats["under_pred_count"] += float(row["under_pred_count"])
    stats["over_pred_count"] += float(row["over_pred_count"])
    stats["target_sum"] += float(row["target_mean"]) * node_count
    stats["target_sq_sum"] += float(row["target_sq_mean"]) * node_count
    stats["pred_sum"] += float(row["pred_mean"]) * node_count
    stats["bias_sum"] += float(row["bias"]) * node_count
    stats["peak_relative_error_sum"] += float(row["peak_relative_error"])
    stats["sample_count"] += 1.0


def _finalize_summary(summary: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, region), stats in sorted(summary.items()):
        node_count = max(stats["node_count"], 1.0)
        sample_count = max(stats["sample_count"], 1.0)
        within_ratio = stats["within25_count"] / node_count
        target_sum = stats["target_sum"]
        target_sq_sum = stats["target_sq_sum"]
        sst = target_sq_sum - (target_sum * target_sum / node_count)
        r2 = None if sst <= 1e-12 else 1.0 - stats["squared_error_sum"] / sst
        rows.append(
            {
                "split": split,
                "region": region,
                "node_count": int(stats["node_count"]),
                "sample_count": int(stats["sample_count"]),
                "mae": stats["abs_error_sum"] / node_count,
                "rmse": math.sqrt(stats["squared_error_sum"] / node_count),
                "within25_count": int(stats["within25_count"]),
                "within25_ratio": within_ratio,
                "miss25_rate": 1.0 - within_ratio,
                "mean_relative_error": stats["relative_error_sum"] / node_count,
                "mean_symmetric_relative_error": stats["symmetric_relative_error_sum"] / node_count,
                "mean_sample_median_relative_error": stats["sample_median_relative_error_weighted_sum"] / node_count,
                "mean_sample_p90_relative_error": stats["sample_p90_relative_error_weighted_sum"] / node_count,
                "mean_sample_p95_relative_error": stats["sample_p95_relative_error_weighted_sum"] / node_count,
                "under_pred_count": int(stats["under_pred_count"]),
                "under_pred_ratio": stats["under_pred_count"] / node_count,
                "over_pred_count": int(stats["over_pred_count"]),
                "over_pred_ratio": stats["over_pred_count"] / node_count,
                "target_mean": stats["target_sum"] / node_count,
                "pred_mean": stats["pred_sum"] / node_count,
                "bias": stats["bias_sum"] / node_count,
                "peak_relative_error": stats["peak_relative_error_sum"] / sample_count,
                "r2": r2,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_split_paths(config: dict[str, Any]) -> dict[str, list[Path]]:
    dataset_cfg = config["dataset"]
    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    split_paths: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        case_dirs = [case_index[name] for name in split_names.get(split, [])]
        split_paths[split] = expand_case_sample_paths(case_dirs, dataset_cfg)
    return split_paths


def _load_prepared_case(
    sample_path: Path,
    config: dict[str, Any],
    scalers: dict[str, StandardScaler],
    two_stage_cfg: dict[str, Any],
) -> PreparedCase:
    dataset_cfg = config["dataset"]
    feature_cfg = dict(config.get("features", {}))
    case = load_case_graph(
        sample_path,
        node_columns=dataset_cfg["node_columns"],
        edge_columns=dataset_cfg["edge_columns"],
        target_freq_key=dataset_cfg.get("target_freq_key", "freq_top3"),
        make_undirected=bool(dataset_cfg.get("make_undirected", True)),
        cache_dir=dataset_cfg.get("cache_dir"),
        mode_shape_loader_kwargs=get_mode_shape_loader_kwargs(feature_cfg),
    )
    return prepare_case(
        case,
        node_scaler=scalers["node"],
        edge_scaler=scalers["edge"],
        global_scaler=scalers["global"],
        target_scaler=scalers["target"],
        task=config.get("task", "field"),
        use_psd=bool(feature_cfg["use_psd"]),
        use_freq_top3=bool(feature_cfg.get("use_freq_top3", False)),
        use_frequency_scalar=bool(feature_cfg.get("use_frequency_scalar", False)),
        use_frequency_relations=bool(feature_cfg.get("use_frequency_relations", False)),
        clamp_negative_rmises=bool(dataset_cfg.get("clamp_negative_rmises", True)),
        two_stage_rmises_cfg=two_stage_cfg,
        feature_cfg=feature_cfg,
        node_columns=dataset_cfg["node_columns"],
        edge_columns=dataset_cfg["edge_columns"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stress predictions by coordinate-derived geometry regions.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--device", type=str, default=None, help="Override evaluation device.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for region metrics.")
    parser.add_argument("--ear-hole-factor", type=float, default=4.0)
    parser.add_argument("--plate-hole-factor", type=float, default=4.0)
    parser.add_argument("--root-band-mm", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    config["training"] = dict(config["training"])
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / "region_eval")

    set_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device or config["training"]["device"])
    scalers = {
        name: StandardScaler.from_state_dict(state)
        for name, state in checkpoint["scalers"].items()
    }
    two_stage_cfg = get_two_stage_rmises_cfg(config)
    stress_peak_relative_cfg = get_stress_peak_relative_cfg(config)
    hotspot_metric_cfg = get_stress_hotspot_metric_cfg(config)
    split_paths = _resolve_split_paths(config)
    splits = list(split_paths) if args.split == "all" else [args.split]
    node_columns = list(config["dataset"]["node_columns"])
    within_relative_error = float(hotspot_metric_cfg.get("within_relative_error", 0.25))
    target_scaler = scalers["target"].to(device)

    sample_case = _load_prepared_case(split_paths["train"][0], config, scalers, two_stage_cfg)
    model = build_model(config, sample_case=sample_case).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sample_rows: list[dict[str, Any]] = []
    summary_accumulator: dict[tuple[str, str], dict[str, float]] = {}

    with torch.no_grad():
        for split in splits:
            for sample_path in split_paths[split]:
                batch = _load_prepared_case(sample_path, config, scalers, two_stage_cfg).to(device)
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
                    target_scaler=target_scaler,
                    clamp_negative_rmises=bool(config["dataset"].get("clamp_negative_rmises", True)),
                    two_stage_rmises_cfg=two_stage_cfg,
                    stress_peak_relative_cfg=stress_peak_relative_cfg,
                )
                stress_pred = prediction_raw.detach().cpu()[:, 0].clamp_min(0.0)
                stress_target = batch.target_metric.detach().cpu()[:, 0].clamp_min(0.0)
                raw_features = _raw_node_features(batch.node_features, scalers["node"])
                payload = _load_global_payload(sample_path)
                masks = _region_masks(
                    raw_node_features=raw_features,
                    node_columns=node_columns,
                    payload=payload,
                    ear_hole_factor=float(args.ear_hole_factor),
                    plate_hole_factor=float(args.plate_hole_factor),
                    root_band_mm=args.root_band_mm,
                )
                case_name = _case_name_from_sample_path(sample_path)
                for region, mask in masks.items():
                    row = _metric_row(
                        split=split,
                        sample=batch.name,
                        case_name=case_name,
                        frequency_hz=batch.frequency_hz,
                        region=region,
                        stress_target=stress_target,
                        stress_pred=stress_pred,
                        mask=mask,
                        within_relative_error=within_relative_error,
                    )
                    if row is None:
                        continue
                    sample_rows.append(row)
                    _accumulate(summary_accumulator, row)

    summary_rows = _finalize_summary(summary_accumulator)
    _write_csv(output_dir / "region_sample_metrics.csv", sample_rows, REGION_FIELDNAMES)
    _write_csv(
        output_dir / "region_summary_metrics.csv",
        summary_rows,
        SUMMARY_FIELDNAMES,
    )
    write_json(output_dir / "region_summary_metrics.json", {"checkpoint": str(checkpoint_path), "rows": summary_rows})
    print(json.dumps({"checkpoint": str(checkpoint_path), "rows": summary_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

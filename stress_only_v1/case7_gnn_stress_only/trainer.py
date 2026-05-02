from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import csv
import json
import math
import random

import torch
import torch.nn.functional as F

from case7_gnn_stress_only.data import (
    build_global_features,
    discover_case_index,
    expand_case_sample_paths,
    load_case_graph,
    resolve_case_splits,
)
from case7_gnn_stress_only.models import FieldGNN, FrequencyGNN
from case7_gnn_stress_only.runtime import ensure_dir, make_logger, write_json, write_yaml
from case7_gnn_stress_only.scalers import (
    RunningTensorStats,
    StandardScaler,
    build_rmises_hotspot_targets,
    decode_field_targets,
    encode_field_targets,
    metric_field_targets,
)


@dataclass
class PreparedCase:
    name: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_bc_mask: torch.Tensor
    edge_distance: torch.Tensor
    global_features: torch.Tensor
    target_normalized: torch.Tensor
    target_metric: torch.Tensor

    def to(self, device: torch.device) -> "PreparedCase":
        return PreparedCase(
            name=self.name,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_bc_mask=self.node_bc_mask.to(device),
            edge_distance=self.edge_distance.to(device),
            global_features=self.global_features.to(device),
            target_normalized=self.target_normalized.to(device),
            target_metric=self.target_metric.to(device),
        )


def get_two_stage_rmises_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = dict(config.get("stress_two_stage", config.get("rmises_two_stage", {})))
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "threshold": float(raw_cfg.get("threshold", 25.0)),
        "threshold_quantile": raw_cfg.get("threshold_quantile"),
        "prob_threshold": float(raw_cfg.get("prob_threshold", 0.5)),
        "classification_weight": float(raw_cfg.get("classification_weight", 1.0)),
        "regression_weight": float(raw_cfg.get("regression_weight", 2.0)),
        "background_regression_weight": float(raw_cfg.get("background_regression_weight", 0.05)),
        "positive_class_weight": float(raw_cfg.get("positive_class_weight", 10.0)),
        "within_relative_error": float(raw_cfg.get("within_relative_error", 0.25)),
    }


def _column_index(column_names: list[str] | tuple[str, ...] | Any, target_name: str) -> int | None:
    names = list(column_names)
    try:
        return names.index(target_name)
    except ValueError:
        return None


def _extract_node_bc_mask(case: Any, node_columns: list[str] | tuple[str, ...] | Any) -> torch.Tensor:
    bc_index = _column_index(node_columns, "bc_mask")
    if bc_index is None:
        return torch.zeros(case.node_features.size(0), dtype=case.node_features.dtype)
    return case.node_features[:, bc_index]


def _extract_edge_distance(case: Any, edge_columns: list[str] | tuple[str, ...] | Any) -> torch.Tensor:
    dist_index = _column_index(edge_columns, "dist")
    if dist_index is None:
        return torch.ones(case.edge_features.size(0), dtype=case.edge_features.dtype)
    return case.edge_features[:, dist_index].abs()


def _infer_earpiece_centers(
    case: Any,
    node_columns: list[str] | tuple[str, ...] | Any,
    feature_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    cfg = feature_cfg or {}
    x_index = _column_index(node_columns, "x")
    y_index = _column_index(node_columns, "y")
    z_index = _column_index(node_columns, "z")
    bc_index = _column_index(node_columns, "bc_mask")
    if x_index is None or y_index is None or z_index is None:
        raise ValueError("earpiece center inference requires x, y, z in dataset.node_columns")

    if bc_index is None:
        center = case.node_features.new_tensor(
            [[float(case.params[1]), 0.0, float(case.params[3])]],
        )
        return center

    bc_mask = case.node_features[:, bc_index] >= 0.5
    bc_points = case.node_features[bc_mask][:, [x_index, y_index, z_index]]
    if bc_points.size(0) == 0:
        center = case.node_features.new_tensor(
            [[float(case.params[1]), 0.0, float(case.params[3])]],
        )
        return center

    theta = torch.atan2(bc_points[:, 1], bc_points[:, 0])
    theta = torch.remainder(theta, 2.0 * math.pi)
    theta_sorted, order = torch.sort(theta)
    points_sorted = bc_points[order]

    wrapped_theta = torch.cat([theta_sorted, theta_sorted[:1] + 2.0 * math.pi], dim=0)
    theta_gap_threshold = float(cfg.get("earpiece_center_gap_threshold_deg", 25.0)) * math.pi / 180.0
    gap_mask = (wrapped_theta[1:] - wrapped_theta[:-1]) > theta_gap_threshold

    split_indices = torch.nonzero(gap_mask, as_tuple=False).flatten().tolist()
    if not split_indices:
        return points_sorted.mean(dim=0, keepdim=True)

    centers: list[torch.Tensor] = []
    start = 0
    for split_index in split_indices:
        end = split_index + 1
        if end > start:
            centers.append(points_sorted[start:end].mean(dim=0))
        start = end
    if start < points_sorted.size(0):
        centers.append(points_sorted[start:].mean(dim=0))

    # Merge first/last clusters if the wrap-around gap was not a true separator.
    if len(centers) > 1 and not bool(gap_mask[-1].item()):
        merged = torch.cat([points_sorted[start:], points_sorted[: split_indices[0] + 1]], dim=0).mean(dim=0)
        middle_points = []
        prev = split_indices[0] + 1
        for split_index in split_indices[1:]:
            middle_points.append(points_sorted[prev : split_index + 1].mean(dim=0))
            prev = split_index + 1
        centers = [merged] + middle_points

    return torch.stack(centers, dim=0)


def build_augmented_node_features(
    case: Any,
    node_columns: list[str] | tuple[str, ...] | Any,
    feature_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    cfg = feature_cfg or {}
    if not bool(cfg.get("augment_high_reliability_features", cfg.get("augment_node_physics", False))):
        return case.node_features

    x_index = _column_index(node_columns, "x")
    y_index = _column_index(node_columns, "y")
    z_index = _column_index(node_columns, "z")
    if x_index is None or y_index is None or z_index is None:
        raise ValueError("augment_node_physics requires x, y, z in dataset.node_columns")

    x = case.node_features[:, x_index : x_index + 1]
    y = case.node_features[:, y_index : y_index + 1]
    z = case.node_features[:, z_index : z_index + 1]

    fixed_geometry = getattr(case, "fixed_geometry", {})
    plate_radius = case.params[6].clamp_min(1e-6)
    plate_thickness = case.node_features.new_tensor(
        float(fixed_geometry.get("plate_thickness", 15.0)),
    ).clamp_min(1e-6)
    earpiece_radial_dist = case.params[1].clamp_min(1e-6)
    earpiece_width = case.params[2].clamp_min(1e-6)
    earpiece_hole_radius = case.node_features.new_tensor(
        float(fixed_geometry.get("earpiece_HoleRadius", 4.0)),
    ).clamp_min(1e-6)
    earpiece_count = max(1, int(float(fixed_geometry.get("earpiece_Count_default", 3))))
    mass_couple_radius = case.node_features.new_tensor(
        float(fixed_geometry.get("mass_couple_radius", 65.0)),
    ).clamp_min(1e-6)

    r = torch.sqrt(x.pow(2) + y.pow(2) + 1e-12)
    theta = torch.atan2(y, x)
    x_norm = x / plate_radius
    y_norm = y / plate_radius
    z_norm = z / plate_thickness
    r_norm = r / plate_radius
    dist_to_edge = (plate_radius - r) / plate_radius

    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        steps=earpiece_count + 1,
        device=case.node_features.device,
        dtype=case.node_features.dtype,
    )[:-1]
    centers = torch.stack(
        [
            -earpiece_radial_dist * torch.sin(angles),
            earpiece_radial_dist * torch.cos(angles),
        ],
        dim=-1,
    )
    node_xy = torch.cat([x, y], dim=-1)
    dist_to_earpiece_center = torch.cdist(node_xy, centers).min(dim=1, keepdim=True).values
    dist_to_ear_hole_edge = dist_to_earpiece_center - earpiece_hole_radius
    dist_to_ear_hole_edge_local = dist_to_ear_hole_edge / earpiece_hole_radius
    dist_to_ear_hole_edge_global = dist_to_ear_hole_edge / plate_radius
    near_ear_hole = torch.exp(-dist_to_ear_hole_edge.clamp_min(0.0) / earpiece_hole_radius)

    center_radius_local = r / mass_couple_radius
    center_couple_signed = (mass_couple_radius - r) / mass_couple_radius
    near_center_couple = torch.sigmoid(5.0 * center_couple_signed)

    physics_features = [
        x_norm,
        y_norm,
        z_norm,
        r_norm,
        dist_to_edge,
        torch.sin(theta),
        torch.cos(theta),
        earpiece_width / plate_radius * torch.ones_like(x),
        earpiece_width / earpiece_hole_radius * torch.ones_like(x),
        dist_to_ear_hole_edge_local,
        dist_to_ear_hole_edge_global,
        near_ear_hole,
        center_radius_local,
        center_couple_signed,
        near_center_couple,
    ]
    return torch.cat([case.node_features] + physics_features, dim=-1)


def decode_field_prediction(
    prediction: torch.Tensor,
    target_scaler: StandardScaler,
    clamp_negative_rmises: bool,
    two_stage_rmises_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if two_stage_rmises_cfg["enabled"]:
        prediction_encoded = target_scaler.inverse_transform(prediction[:, 1:2])
        prediction_raw = decode_field_targets(
            prediction_encoded,
            clamp_negative_rmises=clamp_negative_rmises,
            rmises_as_excess=False,
        )
        hotspot_prob = torch.sigmoid(prediction[:, 0])
        hotspot_mask = hotspot_prob >= float(two_stage_rmises_cfg["prob_threshold"])
        return prediction_raw, hotspot_prob, hotspot_mask

    prediction_encoded = target_scaler.inverse_transform(prediction)
    prediction_raw = decode_field_targets(
        prediction_encoded,
        clamp_negative_rmises=clamp_negative_rmises,
    )
    return prediction_raw, None, None


def prepare_case(
    case: Any,
    node_scaler: StandardScaler,
    edge_scaler: StandardScaler,
    global_scaler: StandardScaler,
    target_scaler: StandardScaler,
    task: str,
    use_psd: bool,
    use_freq_top3: bool,
    use_frequency_scalar: bool,
    use_frequency_relations: bool,
    clamp_negative_rmises: bool,
    node_columns: list[str] | tuple[str, ...] | Any,
    edge_columns: list[str] | tuple[str, ...] | Any,
    feature_cfg: dict[str, Any] | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> PreparedCase:
    two_stage_cfg = two_stage_rmises_cfg or {"enabled": False, "threshold": 0.0}
    global_features = global_scaler.transform(
        build_global_features(
            case,
            use_psd=use_psd,
            use_freq_top3=use_freq_top3,
            use_frequency_scalar=use_frequency_scalar,
            use_frequency_relations=use_frequency_relations,
        )
    )
    node_features = node_scaler.transform(
        build_augmented_node_features(case, node_columns=node_columns, feature_cfg=feature_cfg)
    )
    edge_features = edge_scaler.transform(case.edge_features)
    node_bc_mask = _extract_node_bc_mask(case, node_columns=node_columns)
    edge_distance = _extract_edge_distance(case, edge_columns=edge_columns)

    if task == "frequency":
        target_metric = case.freq_target
        target_normalized = target_scaler.transform(case.freq_target)
    elif task == "field":
        encoded_target = encode_field_targets(
            case.node_targets,
            clamp_negative_rmises=clamp_negative_rmises,
            rmises_as_excess=False,
            rmises_threshold=0.0,
        )
        target_metric = metric_field_targets(case.node_targets, clamp_negative_rmises=clamp_negative_rmises)
        target_normalized = target_scaler.transform(encoded_target)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return PreparedCase(
        name=case.name,
        node_features=node_features,
        edge_index=case.edge_index,
        edge_features=edge_features,
        node_bc_mask=node_bc_mask,
        edge_distance=edge_distance,
        global_features=global_features,
        target_normalized=target_normalized,
        target_metric=target_metric,
    )


def fit_feature_scalers(
    train_case_paths: list[Path],
    dataset_cfg: dict[str, Any],
    task: str,
    use_psd: bool,
    use_freq_top3: bool,
    use_frequency_scalar: bool,
    use_frequency_relations: bool,
    clamp_negative_rmises: bool,
    feature_cfg: dict[str, Any] | None = None,
    case_limit: int | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]:
    two_stage_cfg = two_stage_rmises_cfg or {"enabled": False, "threshold": 0.0}
    selected_paths = list(train_case_paths)
    if case_limit is not None:
        case_limit_int = int(case_limit)
        if case_limit_int <= 0:
            raise ValueError("scaler_fit_case_limit must be positive when provided.")
        selected_paths = selected_paths[:case_limit_int]

    if not selected_paths:
        raise ValueError("Training split is empty; cannot fit feature scalers.")

    cache_dir = dataset_cfg.get("cache_dir")
    node_stats = RunningTensorStats()
    edge_stats = RunningTensorStats()
    global_stats = RunningTensorStats()
    target_stats = RunningTensorStats()

    for case_path in selected_paths:
        case = load_case_graph(
            case_path,
            node_columns=dataset_cfg["node_columns"],
            edge_columns=dataset_cfg["edge_columns"],
            target_freq_key=dataset_cfg["target_freq_key"],
            make_undirected=bool(dataset_cfg["make_undirected"]),
            cache_dir=cache_dir,
        )

        node_stats.update(
            build_augmented_node_features(
                case,
                node_columns=dataset_cfg["node_columns"],
                feature_cfg=feature_cfg,
            )
        )
        edge_stats.update(case.edge_features)
        global_stats.update(
            build_global_features(
                case,
                use_psd=use_psd,
                use_freq_top3=use_freq_top3,
                use_frequency_scalar=use_frequency_scalar,
                use_frequency_relations=use_frequency_relations,
            )
        )

        if task == "frequency":
            target_stats.update(case.freq_target)
        elif task == "field":
            target_stats.update(
                encode_field_targets(
                    case.node_targets,
                    clamp_negative_rmises=clamp_negative_rmises,
                    rmises_as_excess=False,
                    rmises_threshold=0.0,
                )
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    return (
        node_stats.finalize(),
        edge_stats.finalize(),
        global_stats.finalize(),
        target_stats.finalize(),
    )


def build_model(config: dict[str, Any], sample_case: PreparedCase) -> torch.nn.Module:
    task = config["task"]
    model_cfg = config["model"]
    conditioning_cfg = dict(model_cfg.get("conditioning", {}))
    conditioning_enabled = bool(conditioning_cfg.get("enabled", False))
    two_stage_rmises_cfg = get_two_stage_rmises_cfg(config)

    common_kwargs = dict(
        node_input_dim=int(sample_case.node_features.size(-1)),
        edge_input_dim=int(sample_case.edge_features.size(-1)),
        global_input_dim=int(sample_case.global_features.numel()),
        hidden_dim=int(model_cfg["hidden_dim"]),
        global_dim=int(model_cfg["global_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        conditioning_dim=int(conditioning_cfg.get("case_dim", model_cfg["global_dim"])),
        enable_conditioning=conditioning_enabled,
    )

    if task == "frequency":
        return FrequencyGNN(output_dim=int(sample_case.target_normalized.numel()), **common_kwargs)
    if task == "field":
        return FieldGNN(
            output_dim=2 if bool(two_stage_rmises_cfg["enabled"]) else int(sample_case.target_normalized.size(-1)),
            rmises_refine_layers=int(model_cfg.get("rmises_refine_layers", max(1, int(model_cfg["num_layers"]) - 1))),
            use_two_stage_rmises=bool(two_stage_rmises_cfg["enabled"]),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported task: {task}")


def compute_pointwise_loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return (prediction - target).pow(2)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="none")
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_stress_hotspot_targets(stress_values: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    threshold = float(cfg.get("threshold", 0.0))
    threshold_quantile = cfg.get("threshold_quantile")
    if threshold_quantile is not None:
        q = float(threshold_quantile)
        if 0.0 < q < 1.0 and stress_values.numel() > 0:
            threshold = max(threshold, float(torch.quantile(stress_values, q).item()))
    return build_rmises_hotspot_targets(stress_values, threshold=threshold)


def compute_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_metric: torch.Tensor,
    batch: PreparedCase | None,
    loss_name: str,
    field_loss_cfg: dict[str, Any] | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    two_stage_cfg = two_stage_rmises_cfg or {"enabled": False}
    cfg = field_loss_cfg or {}
    stress_channel = 1 if bool(two_stage_cfg["enabled"]) else 0
    physics_stress_smoothness_weight = float(
        cfg.get("physics_stress_smoothness_weight", cfg.get("physics_rmises_smoothness_weight", 0.0))
    )
    physics_distance_power = float(cfg.get("physics_distance_power", 1.0))
    physics_exclude_boundary_edges = bool(cfg.get("physics_exclude_boundary_edges", True))
    physics_hotspot_exempt_quantile = float(cfg.get("physics_hotspot_exempt_quantile", 0.98))

    physics_loss = prediction.new_zeros(())
    if batch is not None and physics_stress_smoothness_weight > 0.0:
        src, dst = batch.edge_index
        edge_mask = torch.ones(src.size(0), dtype=torch.bool, device=prediction.device)
        if physics_exclude_boundary_edges:
            non_boundary = batch.node_bc_mask < 0.5
            edge_mask = edge_mask & non_boundary[src] & non_boundary[dst]

        stress_metric = target_metric[:, 0].clamp_min(0.0)
        if 0.0 < physics_hotspot_exempt_quantile < 1.0 and stress_metric.numel() > 0:
            hotspot_threshold = torch.quantile(stress_metric, physics_hotspot_exempt_quantile)
            hotspot_nodes = stress_metric >= hotspot_threshold
            edge_mask = edge_mask & ~(hotspot_nodes[src] | hotspot_nodes[dst])

        if edge_mask.any():
            edge_distance = batch.edge_distance.clamp_min(1e-6).pow(physics_distance_power)
            edge_norm = edge_distance[edge_mask]
            stress_edge_residual = (prediction[src, stress_channel] - prediction[dst, stress_channel]).pow(2)
            physics_loss = physics_loss + physics_stress_smoothness_weight * (
                stress_edge_residual[edge_mask] / edge_norm
            ).mean()

    stress_loss_weight = float(cfg.get("stress_loss_weight", cfg.get("rmises_loss_weight", 1.0)))
    stress_hotspot_alpha = float(cfg.get("stress_hotspot_alpha", cfg.get("rmises_hotspot_alpha", 0.0)))
    stress_hotspot_gamma = float(cfg.get("stress_hotspot_gamma", cfg.get("rmises_hotspot_gamma", 2.0)))
    stress_noise_floor = float(cfg.get("stress_noise_floor", cfg.get("rmises_noise_floor", 0.0)))
    stress_low_value_weight = float(cfg.get("stress_low_value_weight", cfg.get("rmises_low_value_weight", 1.0)))
    stress_hotspot_quantile = float(cfg.get("stress_hotspot_quantile", cfg.get("rmises_hotspot_quantile", 0.0)))
    stress_hotspot_boost = float(cfg.get("stress_hotspot_boost", cfg.get("rmises_hotspot_boost", 0.0)))
    stress_topk_ratio = float(cfg.get("stress_topk_ratio", cfg.get("rmises_topk_ratio", 0.0)))
    stress_topk_weight = float(cfg.get("stress_topk_weight", cfg.get("rmises_topk_weight", 0.0)))
    stress_peak_weight = float(cfg.get("stress_peak_weight", 0.0))
    stress_case_activity_quantile = float(
        cfg.get("stress_case_activity_quantile", cfg.get("rmises_case_activity_quantile", 0.0))
    )
    stress_case_activity_reference = float(
        cfg.get("stress_case_activity_reference", cfg.get("rmises_case_activity_reference", 1.0))
    )
    stress_case_activity_power = float(
        cfg.get("stress_case_activity_power", cfg.get("rmises_case_activity_power", 1.0))
    )
    stress_case_activity_min_weight = float(
        cfg.get("stress_case_activity_min_weight", cfg.get("rmises_case_activity_min_weight", 1.0))
    )
    stress_case_activity_max_weight = float(
        cfg.get("stress_case_activity_max_weight", cfg.get("rmises_case_activity_max_weight", 1.0))
    )

    stress_prediction = prediction[:, stress_channel]
    stress_target = target[:, 0]
    stress_loss = compute_pointwise_loss(stress_prediction, stress_target, loss_name=loss_name)
    stress_values = target_metric[:, 0].clamp_min(0.0)
    stress_log = torch.log1p(stress_values)
    stress_weights = torch.full_like(stress_loss, fill_value=stress_low_value_weight)

    active_mask = stress_values > stress_noise_floor
    if active_mask.any():
        active_log = stress_log[active_mask]
        active_max_log = active_log.max()
        if active_max_log.item() > 0.0:
            hotspot_score = (active_log / active_max_log).pow(stress_hotspot_gamma)
        else:
            hotspot_score = torch.zeros_like(active_log)
        stress_weights[active_mask] = 1.0 + stress_hotspot_alpha * hotspot_score

        if 0.0 < stress_hotspot_quantile < 1.0 and stress_hotspot_boost > 0.0:
            hotspot_threshold = torch.quantile(stress_values[active_mask], stress_hotspot_quantile)
            hotspot_mask = active_mask & (stress_values >= hotspot_threshold)
            stress_weights = stress_weights + hotspot_mask.to(dtype=stress_weights.dtype) * stress_hotspot_boost

    stress_case_weight = 1.0
    if 0.0 < stress_case_activity_quantile < 1.0:
        if active_mask.any():
            case_activity_value = torch.quantile(stress_values[active_mask], stress_case_activity_quantile)
            reference_log = math.log1p(max(stress_case_activity_reference, 1e-12))
            if reference_log > 0.0:
                activity_ratio = torch.log1p(case_activity_value) / reference_log
                stress_case_weight = float(
                    torch.clamp(
                        activity_ratio.pow(stress_case_activity_power),
                        min=stress_case_activity_min_weight,
                        max=stress_case_activity_max_weight,
                    ).item()
                )
            else:
                stress_case_weight = stress_case_activity_max_weight
        else:
            stress_case_weight = max(stress_case_activity_min_weight, 0.0)

    weighted_stress_loss = (stress_loss * stress_weights).sum() / stress_weights.sum().clamp_min(1e-6)
    extra_hotspot_loss = stress_loss.new_zeros(())
    if stress_topk_ratio > 0.0 and stress_topk_weight > 0.0 and active_mask.any():
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        topk_count = min(active_indices.numel(), max(1, int(math.ceil(active_indices.numel() * stress_topk_ratio))))
        _, topk_order = torch.topk(stress_values[active_indices], k=topk_count, largest=True, sorted=False)
        topk_indices = active_indices[topk_order]
        extra_hotspot_loss = stress_loss[topk_indices].mean() * stress_topk_weight

    peak_loss = stress_loss.new_zeros(())
    if stress_peak_weight > 0.0 and stress_prediction.numel() > 0:
        peak_loss = compute_pointwise_loss(
            stress_prediction.max().reshape(1),
            stress_target.max().reshape(1),
            loss_name=loss_name,
        ).mean() * stress_peak_weight

    if bool(two_stage_cfg["enabled"]):
        hotspot_target = build_stress_hotspot_targets(stress_values, two_stage_cfg)
        pos_weight = prediction.new_tensor(float(two_stage_cfg["positive_class_weight"]))
        hotspot_cls_loss = F.binary_cross_entropy_with_logits(
            prediction[:, 0],
            hotspot_target,
            pos_weight=pos_weight,
        )

        hotspot_mask = hotspot_target > 0.5
        if hotspot_mask.any():
            hotspot_reg_loss = compute_pointwise_loss(
                stress_prediction[hotspot_mask],
                stress_target[hotspot_mask],
                loss_name=loss_name,
            ).mean()
        else:
            hotspot_reg_loss = prediction.new_zeros(())
        background_mask = ~hotspot_mask
        if background_mask.any() and float(two_stage_cfg["background_regression_weight"]) > 0.0:
            background_reg_loss = compute_pointwise_loss(
                stress_prediction[background_mask],
                stress_target[background_mask],
                loss_name=loss_name,
            ).mean() * float(two_stage_cfg["background_regression_weight"])
        else:
            background_reg_loss = prediction.new_zeros(())

        return (
            float(two_stage_cfg["classification_weight"]) * hotspot_cls_loss
            + float(two_stage_cfg["regression_weight"]) * hotspot_reg_loss
            + background_reg_loss
            + stress_loss_weight * stress_case_weight * (weighted_stress_loss + extra_hotspot_loss)
            + peak_loss
            + physics_loss
        )
    return stress_loss_weight * stress_case_weight * (weighted_stress_loss + extra_hotspot_loss) + peak_loss + physics_loss


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    task: str = "frequency",
    target_metric: torch.Tensor | None = None,
    batch: PreparedCase | None = None,
    field_loss_cfg: dict[str, Any] | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    if task == "field":
        if target_metric is None:
            raise ValueError("Field loss requires target_metric for hotspot weighting.")
        return compute_field_loss(
            prediction,
            target,
            target_metric=target_metric,
            batch=batch,
            loss_name=loss_name,
            field_loss_cfg=field_loss_cfg,
            two_stage_rmises_cfg=two_stage_rmises_cfg,
        )
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {loss_name}")


def evaluate_frequency(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    target_scaler: StandardScaler,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    scaler = target_scaler.to(device)
    total_loss = 0.0
    total_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    value_count = 0

    model.eval()
    with torch.no_grad():
        for case_path in case_paths:
            batch = case_loader(case_path).to(device)
            prediction = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                batch.global_features,
            )
            loss = compute_loss(prediction, batch.target_normalized, loss_name=loss_name)
            total_loss += loss.item()
            total_count += 1

            prediction_hz = scaler.inverse_transform(prediction).detach().cpu()
            target_hz = batch.target_metric.detach().cpu()
            error = prediction_hz - target_hz
            abs_sum += error.abs().sum().item()
            sq_sum += error.pow(2).sum().item()
            value_count += int(target_hz.numel())

    denom = max(total_count, 1)
    metric_denom = max(value_count, 1)
    return {
        "loss": total_loss / denom,
        "mae_hz": abs_sum / metric_denom,
        "rmse_hz": (sq_sum / metric_denom) ** 0.5,
    }


def evaluate_field(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    target_scaler: StandardScaler,
    clamp_negative_rmises: bool,
    device: torch.device,
    loss_name: str,
    field_loss_cfg: dict[str, Any] | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    two_stage_cfg = two_stage_rmises_cfg or {"enabled": False}
    scaler = target_scaler.to(device)
    total_loss = 0.0
    total_nodes = 0
    total_stress_abs = 0.0
    total_hotspot_tp = 0.0
    total_hotspot_fp = 0.0
    total_hotspot_fn = 0.0
    total_hotspot_abs = 0.0
    total_hotspot_nodes = 0
    total_hotspot_within_tolerance = 0
    total_top1_abs = 0.0
    total_top1_nodes = 0
    total_top5_abs = 0.0
    total_top5_nodes = 0
    total_peak_relative_error = 0.0
    total_cases = 0

    model.eval()
    with torch.no_grad():
        for case_path in case_paths:
            batch = case_loader(case_path).to(device)
            prediction = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                batch.global_features,
            )
            loss = compute_loss(
                prediction,
                batch.target_normalized,
                loss_name=loss_name,
                task="field",
                target_metric=batch.target_metric,
                batch=batch,
                field_loss_cfg=field_loss_cfg,
                two_stage_rmises_cfg=two_stage_cfg,
            )
            node_count = int(batch.node_features.size(0))
            total_loss += loss.item() * node_count
            total_nodes += node_count

            prediction_raw, hotspot_prob, hotspot_mask = decode_field_prediction(
                prediction=prediction,
                target_scaler=scaler,
                clamp_negative_rmises=clamp_negative_rmises,
                two_stage_rmises_cfg=two_stage_cfg,
            )
            prediction_raw = prediction_raw.detach().cpu()
            target_raw = batch.target_metric.detach().cpu()
            error = (prediction_raw - target_raw).abs()
            stress_target = target_raw[:, 0].clamp_min(0.0)
            stress_error = error[:, 0]
            total_stress_abs += stress_error.sum().item()
            total_cases += 1

            if stress_target.numel() > 0:
                target_peak = stress_target.max().item()
                pred_peak = prediction_raw[:, 0].clamp_min(0.0).max().item()
                total_peak_relative_error += abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12)

                top1_threshold = torch.quantile(stress_target, 0.99)
                top1_mask = stress_target >= top1_threshold
                total_top1_abs += stress_error[top1_mask].sum().item()
                total_top1_nodes += int(top1_mask.sum().item())

                top5_threshold = torch.quantile(stress_target, 0.95)
                top5_mask = stress_target >= top5_threshold
                total_top5_abs += stress_error[top5_mask].sum().item()
                total_top5_nodes += int(top5_mask.sum().item())

            if bool(two_stage_cfg["enabled"]):
                assert hotspot_prob is not None and hotspot_mask is not None
                target_hotspot = build_stress_hotspot_targets(stress_target, two_stage_cfg).to(dtype=torch.bool)
                pred_hotspot = hotspot_mask.detach().cpu().to(dtype=torch.bool)
                total_hotspot_tp += float((pred_hotspot & target_hotspot).sum().item())
                total_hotspot_fp += float((pred_hotspot & ~target_hotspot).sum().item())
                total_hotspot_fn += float((~pred_hotspot & target_hotspot).sum().item())
                if target_hotspot.any():
                    total_hotspot_abs += stress_error[target_hotspot].sum().item()
                    hotspot_count = int(target_hotspot.sum().item())
                    total_hotspot_nodes += hotspot_count
                    hotspot_relative_error = stress_error[target_hotspot] / stress_target[target_hotspot].abs().clamp_min(1e-12)
                    total_hotspot_within_tolerance += int(
                        (hotspot_relative_error <= float(two_stage_cfg["within_relative_error"])).sum().item()
                    )

    denom = max(total_nodes, 1)
    metrics = {
        "loss": total_loss / denom,
        "stress_mae": total_stress_abs / denom,
        "stress_top1_mae": total_top1_abs / max(total_top1_nodes, 1),
        "stress_top5_mae": total_top5_abs / max(total_top5_nodes, 1),
        "stress_peak_relative_error": total_peak_relative_error / max(total_cases, 1),
    }
    if bool(two_stage_cfg["enabled"]):
        precision = total_hotspot_tp / max(total_hotspot_tp + total_hotspot_fp, 1.0)
        recall = total_hotspot_tp / max(total_hotspot_tp + total_hotspot_fn, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        hotspot_within_tolerance_ratio = total_hotspot_within_tolerance / max(total_hotspot_nodes, 1)
        metrics["stress_hotspot_mae"] = total_hotspot_abs / max(total_hotspot_nodes, 1)
        metrics["stress_hotspot_within25_ratio"] = hotspot_within_tolerance_ratio
        metrics["stress_hotspot_miss25_rate"] = 1.0 - hotspot_within_tolerance_ratio
        metrics["hotspot_precision"] = precision
        metrics["hotspot_recall"] = recall
        metrics["hotspot_f1"] = f1
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    grad_clip: float,
    task: str = "frequency",
    field_loss_cfg: dict[str, Any] | None = None,
    two_stage_rmises_cfg: dict[str, Any] | None = None,
) -> float:
    model.train()
    shuffled = list(case_paths)
    random.shuffle(shuffled)

    total_loss = 0.0
    total_weight = 0

    for case_path in shuffled:
        batch = case_loader(case_path).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            batch.global_features,
        )
        loss = compute_loss(
            prediction,
            batch.target_normalized,
            loss_name=loss_name,
            task=task,
            target_metric=batch.target_metric if task == "field" else None,
            batch=batch if task == "field" else None,
            field_loss_cfg=field_loss_cfg,
            two_stage_rmises_cfg=two_stage_rmises_cfg,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        weight = int(batch.node_features.size(0)) if batch.target_normalized.dim() == 2 else 1
        total_loss += loss.item() * weight
        total_weight += weight

    return total_loss / max(total_weight, 1)


def _save_checkpoint(
    save_dir: Path,
    config: dict[str, Any],
    model: torch.nn.Module,
    scalers: dict[str, StandardScaler],
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
        "metrics": metrics,
    }
    torch.save(checkpoint, save_dir / "best.pt")


class Case7Trainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        self.task = config["task"]
        self.dataset_cfg = config["dataset"]
        self.training_cfg = config["training"]
        self.field_loss_cfg = dict(config.get("field_loss", {}))
        self.two_stage_rmises_cfg = get_two_stage_rmises_cfg(config)
        self.feature_cfg = dict(config.get("features", {}))
        self.use_psd = bool(self.feature_cfg["use_psd"])
        self.use_freq_top3 = bool(self.feature_cfg.get("use_freq_top3", False))
        self.use_frequency_scalar = bool(self.feature_cfg.get("use_frequency_scalar", False))
        self.use_frequency_relations = bool(self.feature_cfg.get("use_frequency_relations", False))
        self.clamp_negative_rmises = bool(self.dataset_cfg.get("clamp_negative_rmises", True))
        self.cache_dir = self.dataset_cfg.get("cache_dir")

        self.save_dir = ensure_dir(self.training_cfg["save_dir"])
        self.logger = make_logger(self.save_dir, logger_name=f"case7_gnn_stress_only.{self.task}")
        self.history_path = self.save_dir / "history.csv"

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scalers: dict[str, StandardScaler] = {}
        self.case_index: dict[str, Path] = {}
        self.split_names: dict[str, list[str]] = {}
        self.train_case_paths: list[Path] = []
        self.val_case_paths: list[Path] = []
        self.test_case_paths: list[Path] = []
        self.resolved_config: dict[str, Any] = {}

        self._prepare()

    def _load_prepared_case(self, case_path: Path) -> PreparedCase:
        case = load_case_graph(
            case_path,
            node_columns=self.dataset_cfg["node_columns"],
            edge_columns=self.dataset_cfg["edge_columns"],
            target_freq_key=self.dataset_cfg["target_freq_key"],
            make_undirected=bool(self.dataset_cfg["make_undirected"]),
            cache_dir=self.cache_dir,
        )
        return prepare_case(
            case,
            node_scaler=self.scalers["node"],
            edge_scaler=self.scalers["edge"],
            global_scaler=self.scalers["global"],
            target_scaler=self.scalers["target"],
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            use_frequency_scalar=self.use_frequency_scalar,
            use_frequency_relations=self.use_frequency_relations,
            clamp_negative_rmises=self.clamp_negative_rmises,
            node_columns=self.dataset_cfg["node_columns"],
            edge_columns=self.dataset_cfg["edge_columns"],
            feature_cfg=self.feature_cfg,
            two_stage_rmises_cfg=self.two_stage_rmises_cfg,
        )

    def _prepare(self) -> None:
        self.case_index = discover_case_index(self.dataset_cfg["root"])
        self.split_names = resolve_case_splits(self.dataset_cfg["root"], self.dataset_cfg)

        train_case_dirs = [self.case_index[name] for name in self.split_names["train"]]
        val_case_dirs = [self.case_index[name] for name in self.split_names["val"]]
        test_case_dirs = [self.case_index[name] for name in self.split_names.get("test", [])]

        self.train_case_paths = expand_case_sample_paths(train_case_dirs, self.dataset_cfg)
        self.val_case_paths = expand_case_sample_paths(val_case_dirs, self.dataset_cfg)
        self.test_case_paths = expand_case_sample_paths(test_case_dirs, self.dataset_cfg)

        node_scaler, edge_scaler, global_scaler, target_scaler = fit_feature_scalers(
            train_case_paths=self.train_case_paths,
            dataset_cfg=self.dataset_cfg,
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            use_frequency_scalar=self.use_frequency_scalar,
            use_frequency_relations=self.use_frequency_relations,
            clamp_negative_rmises=self.clamp_negative_rmises,
            feature_cfg=self.feature_cfg,
            case_limit=self.dataset_cfg.get("scaler_fit_case_limit"),
            two_stage_rmises_cfg=self.two_stage_rmises_cfg,
        )

        self.scalers = {
            "node": node_scaler,
            "edge": edge_scaler,
            "global": global_scaler,
            "target": target_scaler,
        }

        sample_case = self._load_prepared_case(self.train_case_paths[0])
        self.model = build_model(self.config, sample_case=sample_case).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg["lr"]),
            weight_decay=float(self.training_cfg["weight_decay"]),
        )

        resolved_dataset = dict(self.dataset_cfg)
        resolved_dataset["train_cases"] = list(self.split_names["train"])
        resolved_dataset["val_cases"] = list(self.split_names["val"])
        resolved_dataset["test_cases"] = list(self.split_names.get("test", []))
        self.resolved_config = dict(self.config)
        self.resolved_config["dataset"] = resolved_dataset
        write_yaml(self.save_dir / "resolved_config.yaml", self.resolved_config)

    def _evaluate(self, case_paths: list[Path], loss_name: str) -> dict[str, float]:
        assert self.model is not None
        target_scaler = self.scalers["target"]
        if self.task == "frequency":
            return evaluate_frequency(
                model=self.model,
                case_paths=case_paths,
                case_loader=self._load_prepared_case,
                target_scaler=target_scaler,
                device=self.device,
                loss_name=loss_name,
            )
        return evaluate_field(
            model=self.model,
            case_paths=case_paths,
            case_loader=self._load_prepared_case,
            target_scaler=target_scaler,
            clamp_negative_rmises=self.clamp_negative_rmises,
            device=self.device,
            loss_name=loss_name,
            field_loss_cfg=self.field_loss_cfg,
            two_stage_rmises_cfg=self.two_stage_rmises_cfg,
        )

    def _append_history_row(self, row: dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        file_exists = self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def fit(self) -> dict[str, Any]:
        assert self.model is not None
        assert self.optimizer is not None

        eval_every = int(self.training_cfg.get("eval_every", 1))
        if eval_every <= 0:
            raise ValueError("training.eval_every must be positive.")

        self.logger.info("Task: %s", self.task)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Validation selection metric: %s", self.training_cfg.get("selection_metric", "loss"))
        self.logger.info(
            "Train/Val/Test graphs: %s/%s/%s",
            len(self.train_case_paths),
            len(self.val_case_paths),
            len(self.test_case_paths),
        )
        self.logger.info("Using PSD features: %s", self.use_psd)
        self.logger.info("Using freq_top3 features: %s", self.use_freq_top3)
        self.logger.info("Using scalar frequency feature: %s", self.use_frequency_scalar)
        self.logger.info("Using frequency relation features: %s", self.use_frequency_relations)
        self.logger.info(
            "High-reliability node feature augmentation: %s",
            bool(self.feature_cfg.get("augment_high_reliability_features", self.feature_cfg.get("augment_node_physics", False))),
        )
        self.logger.info(
            "Case conditioning: %s",
            bool(self.config.get("model", {}).get("conditioning", {}).get("enabled", False)),
        )
        if self.task == "field":
            self.logger.info("Two-stage stress hotspot: %s", bool(self.two_stage_rmises_cfg["enabled"]))
            if self.two_stage_rmises_cfg["enabled"]:
                self.logger.info("Two-stage stress config: %s", json.dumps(self.two_stage_rmises_cfg, ensure_ascii=False))
        if self.task == "field" and self.field_loss_cfg:
            self.logger.info("Field loss config: %s", json.dumps(self.field_loss_cfg, ensure_ascii=False))
        self.logger.info("Split mode: %s", self.dataset_cfg.get("split_mode", "explicit"))
        if self.cache_dir:
            self.logger.info("Raw case cache: %s", self.cache_dir)
        if self.dataset_cfg.get("scaler_fit_case_limit") is not None:
            self.logger.info("Scaler fit case limit: %s", self.dataset_cfg["scaler_fit_case_limit"])

        selection_metric = str(self.training_cfg.get("selection_metric", "loss"))
        best_val_score = float("inf")
        best_payload: dict[str, Any] | None = None
        patience = int(self.training_cfg["early_stopping_patience"])
        wait = 0

        for epoch in range(1, int(self.training_cfg["epochs"]) + 1):
            train_loss = train_one_epoch(
                model=self.model,
                case_paths=self.train_case_paths,
                case_loader=self._load_prepared_case,
                optimizer=self.optimizer,
                device=self.device,
                loss_name=self.training_cfg["loss"],
                grad_clip=float(self.training_cfg["grad_clip"]),
                task=self.task,
                field_loss_cfg=self.field_loss_cfg,
                two_stage_rmises_cfg=self.two_stage_rmises_cfg,
            )

            history_row = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 8),
            }

            if epoch % eval_every != 0:
                self._append_history_row(history_row)
                if epoch == 1 or epoch % int(self.training_cfg["print_every"]) == 0:
                    self.logger.info("Epoch %04d | train_loss=%.6f | eval=skipped", epoch, train_loss)
                continue

            val_metrics = self._evaluate(self.val_case_paths, loss_name=self.training_cfg["loss"])
            test_metrics = self._evaluate(self.test_case_paths, loss_name=self.training_cfg["loss"])

            history_row["val_loss"] = round(float(val_metrics["loss"]), 8)
            history_row["test_loss"] = round(float(test_metrics["loss"]), 8)
            for key, value in val_metrics.items():
                if key != "loss":
                    history_row[f"val_{key}"] = round(float(value), 8)
            for key, value in test_metrics.items():
                if key != "loss":
                    history_row[f"test_{key}"] = round(float(value), 8)
            self._append_history_row(history_row)

            if epoch == 1 or epoch % int(self.training_cfg["print_every"]) == 0:
                self.logger.info(
                    "Epoch %04d | train_loss=%.6f | val=%s | test=%s",
                    epoch,
                    train_loss,
                    json.dumps(val_metrics, ensure_ascii=False),
                    json.dumps(test_metrics, ensure_ascii=False),
                )

            if selection_metric not in val_metrics:
                raise KeyError(f"Validation metric '{selection_metric}' is not available: {sorted(val_metrics)}")
            val_score = float(val_metrics[selection_metric])

            if val_score < best_val_score:
                best_val_score = val_score
                wait = 0
                best_payload = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "selection_metric": selection_metric,
                    "selection_score": val_score,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
                _save_checkpoint(
                    save_dir=self.save_dir,
                    config=self.resolved_config,
                    model=self.model,
                    scalers=self.scalers,
                    metrics=best_payload,
                )
                write_json(self.save_dir / "metrics.json", best_payload)
            else:
                wait += 1
                if wait >= patience:
                    self.logger.info("Early stopping at epoch %s.", epoch)
                    break

        if best_payload is None:
            raise RuntimeError("Training finished without a saved checkpoint.")

        self.logger.info("Best run summary:\n%s", json.dumps(best_payload, indent=2, ensure_ascii=False))
        return best_payload

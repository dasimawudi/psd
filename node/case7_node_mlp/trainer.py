from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import OrderedDict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import csv
import json
import math
import random
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from case7_node_mlp.data import (
    PER_FREQUENCY_TARGET_COLUMN,
    RawPointSample,
    build_node_selection_mask,
    _load_aligned_target_column,
    discover_case_index,
    expand_case_sample_paths,
    load_raw_point_sample,
    resolve_case_splits,
)
from case7_node_mlp.models import PointMLP, regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, write_json, write_yaml
from case7_node_mlp.scalers import RunningTensorStats, StandardScaler


HISTORY_BASE_FIELDS = ["epoch", "train_loss", "val_loss"]


@dataclass
class PreparedPointSample:
    name: str
    case_name: str
    frequency_hz: float
    features: torch.Tensor
    target_scaled: torch.Tensor
    target_log: torch.Tensor
    target_raw: torch.Tensor
    node_indices: torch.Tensor
    point_weights: torch.Tensor
    region_masks: dict[str, torch.Tensor] | None = None

    @property
    def num_points(self) -> int:
        return int(self.target_raw.numel())


@dataclass
class PointBatch:
    names: list[str]
    case_names: list[str]
    frequency_hz: torch.Tensor
    features: torch.Tensor
    target_scaled: torch.Tensor
    target_log: torch.Tensor
    target_raw: torch.Tensor
    sample_index: torch.Tensor
    node_indices: torch.Tensor
    point_weights: torch.Tensor
    region_masks: dict[str, torch.Tensor] | None = None

    @property
    def num_points(self) -> int:
        return int(self.target_raw.numel())

    def to(self, device: torch.device) -> "PointBatch":
        return PointBatch(
            names=self.names,
            case_names=self.case_names,
            frequency_hz=self.frequency_hz.to(device),
            features=self.features.to(device),
            target_scaled=self.target_scaled.to(device),
            target_log=self.target_log.to(device),
            target_raw=self.target_raw.to(device),
            sample_index=self.sample_index.to(device),
            node_indices=self.node_indices.to(device),
            point_weights=self.point_weights.to(device),
            region_masks=(
                {name: mask.to(device) for name, mask in self.region_masks.items()}
                if self.region_masks is not None
                else None
            ),
        )

    def pin_memory(self) -> "PointBatch":
        return PointBatch(
            names=self.names,
            case_names=self.case_names,
            frequency_hz=self.frequency_hz.pin_memory(),
            features=self.features.pin_memory(),
            target_scaled=self.target_scaled.pin_memory(),
            target_log=self.target_log.pin_memory(),
            target_raw=self.target_raw.pin_memory(),
            sample_index=self.sample_index.pin_memory(),
            node_indices=self.node_indices.pin_memory(),
            point_weights=self.point_weights.pin_memory(),
            region_masks=(
                {name: mask.pin_memory() for name, mask in self.region_masks.items()}
                if self.region_masks is not None
                else None
            ),
        )


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    diagnostics: list[dict[str, Any]]


TOPK_WITHIN25_GROUPS = [
    ("top1", 0.0, 0.01),
    ("top1_5", 0.01, 0.05),
    ("top5", 0.0, 0.05),
    ("top5_10", 0.05, 0.10),
    ("top10", 0.0, 0.10),
]


REGION_TOPK_METRIC_SPECS = [
    ("fullpart_stress", "fullpart_region"),
    ("earpiece_region_stress", "earpiece_region"),
    ("disk_stress", "disk_region"),
    ("disk_center_stress", "disk_center_region"),
]


EVALUATION_DIAGNOSTIC_FIELDS = [
    "split",
    "epoch",
    "sample_name",
    "case_name",
    "frequency_hz",
    "points",
    "loss",
    "mae",
    "rmse",
    "log_mae",
    "log_rmse",
    "bias",
    "relative_mae",
    "symmetric_relative_mae",
    "within25_ratio",
    "target_mean",
    "pred_mean",
    "target_peak",
    "pred_peak",
    "peak_relative_error",
    "top1_mae",
    "top1_log_mae",
    "top5_mae",
    "top5_log_mae",
    "top1_within25_ratio",
    "top1_5_within25_ratio",
    "top5_within25_ratio",
    "top5_10_within25_ratio",
    "top10_within25_ratio",
    "under_pred_ratio",
    "over_pred_ratio",
]


def _format_duration(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _empty_point_metric_totals() -> dict[str, float]:
    return {
        "loss_sum": 0.0,
        "weighted_loss_sum": 0.0,
        "weight_sum": 0.0,
        "points": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "log_abs_sum": 0.0,
        "log_sq_sum": 0.0,
        "bias_sum": 0.0,
        "relative_sum": 0.0,
        "symmetric_relative_sum": 0.0,
        "within25_count": 0.0,
        "relative_points": 0.0,
        "under_count": 0.0,
        "over_count": 0.0,
        "target_sum": 0.0,
        "pred_sum": 0.0,
    }


def _update_point_metric_totals(
    totals: dict[str, float],
    *,
    loss_sum: float,
    weighted_loss_sum: float | None = None,
    weight_sum: float | None = None,
    pred_log: torch.Tensor,
    pred_raw: torch.Tensor,
    target_log: torch.Tensor,
    target_raw: torch.Tensor,
) -> None:
    pred_log = pred_log.reshape(-1)
    pred_raw = pred_raw.reshape(-1)
    target_log = target_log.reshape(-1)
    target_raw = target_raw.reshape(-1)
    point_count = int(target_raw.numel())
    if point_count <= 0:
        return

    raw_delta = pred_raw - target_raw
    log_delta = pred_log - target_log
    abs_error = raw_delta.abs()
    log_abs_error = log_delta.abs()
    relative_mask = target_raw.abs() > 1e-12
    relative_error = torch.empty_like(abs_error)
    relative_error[relative_mask] = abs_error[relative_mask] / target_raw.abs()[relative_mask]
    relative_error[~relative_mask] = 0.0
    symmetric_relative_error = abs_error / (0.5 * (pred_raw.abs() + target_raw.abs())).clamp_min(1e-12)

    totals["loss_sum"] += float(loss_sum)
    totals["weighted_loss_sum"] += float(weighted_loss_sum if weighted_loss_sum is not None else loss_sum)
    totals["weight_sum"] += float(weight_sum if weight_sum is not None else point_count)
    totals["points"] += float(point_count)
    totals["abs_sum"] += abs_error.sum().item()
    totals["sq_sum"] += raw_delta.pow(2).sum().item()
    totals["log_abs_sum"] += log_abs_error.sum().item()
    totals["log_sq_sum"] += log_delta.pow(2).sum().item()
    totals["bias_sum"] += raw_delta.sum().item()
    totals["relative_sum"] += relative_error[relative_mask].sum().item()
    totals["symmetric_relative_sum"] += symmetric_relative_error.sum().item()
    totals["within25_count"] += (relative_error[relative_mask] <= 0.25).sum().item()
    totals["relative_points"] += relative_mask.sum().item()
    totals["under_count"] += (pred_raw < target_raw).sum().item()
    totals["over_count"] += (pred_raw > target_raw).sum().item()
    totals["target_sum"] += target_raw.sum().item()
    totals["pred_sum"] += pred_raw.sum().item()


def _finalize_point_metrics(totals: dict[str, float]) -> dict[str, float]:
    points = max(float(totals["points"]), 1.0)
    weight_sum = max(float(totals["weight_sum"]), 1.0)
    relative_points = max(float(totals["relative_points"]), 1.0)
    return {
        "loss": totals["loss_sum"] / points,
        "weighted_loss": totals["weighted_loss_sum"] / weight_sum,
        "earpiece_stress_mae": totals["abs_sum"] / points,
        "earpiece_stress_rmse": math.sqrt(totals["sq_sum"] / points),
        "earpiece_stress_log_mae": totals["log_abs_sum"] / points,
        "earpiece_stress_log_rmse": math.sqrt(totals["log_sq_sum"] / points),
        "earpiece_stress_bias": totals["bias_sum"] / points,
        "earpiece_stress_relative_mae": totals["relative_sum"] / relative_points,
        "earpiece_stress_symmetric_relative_mae": totals["symmetric_relative_sum"] / points,
        "earpiece_stress_within25_ratio": totals["within25_count"] / relative_points,
        "earpiece_stress_miss25_rate": 1.0 - totals["within25_count"] / relative_points,
        "earpiece_stress_under_pred_ratio": totals["under_count"] / points,
        "earpiece_stress_over_pred_ratio": totals["over_count"] / points,
        "earpiece_stress_target_mean": totals["target_sum"] / points,
        "earpiece_stress_pred_mean": totals["pred_sum"] / points,
        "points": float(totals["points"]),
        "relative_points": float(totals["relative_points"]),
    }


def _rank_fraction_masks(values: torch.Tensor) -> dict[str, torch.Tensor]:
    values = values.reshape(-1)
    point_count = int(values.numel())
    if point_count <= 0:
        return {}
    rank_fraction = torch.empty(point_count, dtype=torch.float32, device=values.device)
    order = torch.argsort(torch.nan_to_num(values, nan=-torch.inf), descending=True, stable=True)
    rank_fraction[order] = torch.arange(1, point_count + 1, dtype=torch.float32, device=values.device) / float(
        point_count
    )
    return {
        name: (rank_fraction > float(left)) & (rank_fraction <= float(right))
        for name, left, right in TOPK_WITHIN25_GROUPS
    }


def _sample_rank_fractions(target_raw: torch.Tensor, sample_index: torch.Tensor) -> torch.Tensor:
    target_raw = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)
    rank_fraction = torch.empty_like(target_raw, dtype=torch.float32)
    if target_raw.numel() == 0:
        return rank_fraction

    sample_ids, counts = torch.unique_consecutive(sample_index, return_counts=True)
    if int(counts.sum().item()) != int(sample_index.numel()) or int(sample_ids.numel()) != int(torch.unique(sample_index).numel()):
        sample_ids = torch.unique(sample_index)
        for sample_id in sample_ids:
            mask = sample_index == sample_id
            if not bool(mask.any()):
                continue
            sample_values = target_raw[mask]
            point_count = int(sample_values.numel())
            order = torch.argsort(torch.nan_to_num(sample_values, nan=-torch.inf), descending=True, stable=True)
            sample_rank = torch.empty(point_count, dtype=torch.float32, device=target_raw.device)
            sample_rank[order] = torch.arange(
                1, point_count + 1, dtype=torch.float32, device=target_raw.device
            ) / float(point_count)
            rank_fraction[mask] = sample_rank
        return rank_fraction

    start = 0
    for count in counts.tolist():
        end = start + int(count)
        if end <= start:
            continue
        sample_values = target_raw[start:end]
        point_count = int(sample_values.numel())
        order = torch.argsort(torch.nan_to_num(sample_values, nan=-torch.inf), descending=True, stable=True)
        sample_rank = torch.empty(point_count, dtype=torch.float32, device=target_raw.device)
        sample_rank[order] = torch.arange(1, point_count + 1, dtype=torch.float32, device=target_raw.device) / float(
            point_count
        )
        rank_fraction[start:end] = sample_rank
        start = end
    return rank_fraction


def _empty_topk_metric_totals() -> dict[str, dict[str, float]]:
    return {name: _empty_point_metric_totals() for name, _, _ in TOPK_WITHIN25_GROUPS}


def _update_topk_metric_totals(
    totals_by_group: dict[str, dict[str, float]],
    *,
    pred_log: torch.Tensor,
    pred_raw: torch.Tensor,
    target_log: torch.Tensor,
    target_raw: torch.Tensor,
) -> None:
    for name, mask in _rank_fraction_masks(target_raw).items():
        if not bool(mask.any()):
            continue
        _update_point_metric_totals(
            totals_by_group[name],
            loss_sum=0.0,
            pred_log=pred_log[mask],
            pred_raw=pred_raw[mask],
            target_log=target_log[mask],
            target_raw=target_raw[mask],
        )


def _finalize_topk_metric_totals(totals_by_group: dict[str, dict[str, float]]) -> dict[str, float]:
    return _finalize_topk_metric_totals_with_prefix(totals_by_group, prefix="earpiece_stress")


def _finalize_point_metric_totals_with_prefix(totals: dict[str, float], prefix: str) -> dict[str, float]:
    group_metrics = _finalize_point_metrics(totals)
    field_map = {
        "points": "points",
        "relative_points": "relative_points",
        "mae": "earpiece_stress_mae",
        "rmse": "earpiece_stress_rmse",
        "log_mae": "earpiece_stress_log_mae",
        "log_rmse": "earpiece_stress_log_rmse",
        "bias": "earpiece_stress_bias",
        "relative_mae": "earpiece_stress_relative_mae",
        "symmetric_relative_mae": "earpiece_stress_symmetric_relative_mae",
        "within25_ratio": "earpiece_stress_within25_ratio",
        "miss25_rate": "earpiece_stress_miss25_rate",
        "under_pred_ratio": "earpiece_stress_under_pred_ratio",
        "over_pred_ratio": "earpiece_stress_over_pred_ratio",
        "target_mean": "earpiece_stress_target_mean",
        "pred_mean": "earpiece_stress_pred_mean",
    }
    metrics = {f"{prefix}_{suffix}": float(group_metrics[source]) for suffix, source in field_map.items()}
    target_mean = max(abs(group_metrics["earpiece_stress_target_mean"]), 1e-12)
    metrics[f"{prefix}_pred_target_ratio"] = group_metrics["earpiece_stress_pred_mean"] / target_mean
    return metrics


def _finalize_topk_metric_totals_with_prefix(
    totals_by_group: dict[str, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    field_map = {
        "points": "points",
        "relative_points": "relative_points",
        "mae": "earpiece_stress_mae",
        "log_mae": "earpiece_stress_log_mae",
        "relative_mae": "earpiece_stress_relative_mae",
        "within25_ratio": "earpiece_stress_within25_ratio",
        "miss25_rate": "earpiece_stress_miss25_rate",
        "under_pred_ratio": "earpiece_stress_under_pred_ratio",
        "over_pred_ratio": "earpiece_stress_over_pred_ratio",
        "target_mean": "earpiece_stress_target_mean",
        "pred_mean": "earpiece_stress_pred_mean",
    }
    for name, totals in totals_by_group.items():
        group_metrics = _finalize_point_metrics(totals)
        for suffix, source in field_map.items():
            metrics[f"{prefix}_{name}_{suffix}"] = float(group_metrics[source])
        target_mean = max(abs(group_metrics["earpiece_stress_target_mean"]), 1e-12)
        metrics[f"{prefix}_{name}_pred_target_ratio"] = (
            group_metrics["earpiece_stress_pred_mean"] / target_mean
        )
    return metrics


def _within25_ratio_for_mask(pred_raw: torch.Tensor, target_raw: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    target = target_raw[mask]
    relative_mask = target.abs() > 1e-12
    if not bool(relative_mask.any()):
        return 0.0
    pred = pred_raw[mask]
    relative = (pred[relative_mask] - target[relative_mask]).abs() / target[relative_mask].abs()
    return (relative <= 0.25).to(torch.float32).mean().item()


def _resolve_threshold_from_config(
    target_cfg: dict[str, Any],
    feature_schema: dict[str, Any],
    key: str,
) -> float:
    raw_value = target_cfg.get(key, 0.0)
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"", "none", "off", "false"}:
            return 0.0
        if value.startswith("p99*"):
            p99 = float(feature_schema.get("target_positive_p99", 0.0))
            return p99 * float(value.split("*", 1)[1])
        if value.startswith("p95*"):
            p95 = float(feature_schema.get("target_positive_p95", 0.0))
            return p95 * float(value.split("*", 1)[1])
        return float(value)
    return float(raw_value or 0.0)


def _apply_target_floor(raw_target: torch.Tensor, threshold: float) -> torch.Tensor:
    target = raw_target.clamp_min(0.0)
    if threshold <= 0.0:
        return target
    return torch.where(target < float(threshold), torch.zeros_like(target), target)


def _build_point_weights(
    target_raw: torch.Tensor,
    feature_schema: dict[str, Any],
    loss_cfg: dict[str, Any],
    region_masks: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    weights = torch.ones_like(target_raw, dtype=torch.float32)
    weighting = str(loss_cfg.get("weighting", "none")).lower()
    if weighting == "target_quantile":
        top5_threshold = float(loss_cfg.get("top5_threshold", feature_schema.get("target_positive_p95", 0.0)) or 0.0)
        top1_threshold = float(loss_cfg.get("top1_threshold", feature_schema.get("target_positive_p99", 0.0)) or 0.0)
        top5_weight = float(loss_cfg.get("top5_weight", 2.0))
        top1_weight = float(loss_cfg.get("top1_weight", 5.0))

        if top5_threshold > 0.0 and top5_weight > 1.0:
            weights = torch.where(target_raw >= top5_threshold, torch.full_like(weights, top5_weight), weights)
        if top1_threshold > 0.0 and top1_weight > top5_weight:
            weights = torch.where(target_raw >= top1_threshold, torch.full_like(weights, top1_weight), weights)

    for spec in loss_cfg.get("target_bucket_weights", []) or []:
        if not isinstance(spec, dict):
            continue
        bucket_weight = float(spec.get("weight", 1.0))
        if bucket_weight <= 1.0:
            continue
        min_value = spec.get("min")
        max_value = spec.get("max")
        mask = torch.ones_like(target_raw, dtype=torch.bool)
        if min_value is not None:
            mask &= target_raw >= float(min_value)
        if max_value is not None:
            mask &= target_raw < float(max_value)
        weights = torch.where(mask, torch.maximum(weights, torch.full_like(weights, bucket_weight)), weights)

    center_weight = float(loss_cfg.get("center_region_point_weight", 1.0))
    if center_weight > 1.0 and region_masks is not None and "disk_center_region" in region_masks:
        center_mask = region_masks["disk_center_region"].to(dtype=torch.bool, device=target_raw.device).reshape(-1)
        if bool(center_mask.any()):
            quantile = min(max(float(loss_cfg.get("center_region_top_quantile", 0.95)), 0.0), 1.0)
            min_target = float(loss_cfg.get("center_region_min_target", 0.0))
            center_targets = target_raw.reshape(-1)[center_mask]
            threshold = torch.quantile(center_targets, quantile)
            peak_mask = center_mask & (target_raw.reshape(-1) >= threshold) & (target_raw.reshape(-1) > min_target)
            weights = torch.where(peak_mask, weights * center_weight, weights)

    earpiece_weight = float(loss_cfg.get("earpiece_region_point_weight", 1.0))
    if earpiece_weight > 1.0 and region_masks is not None and "earpiece_region" in region_masks:
        earpiece_mask = region_masks["earpiece_region"].to(dtype=torch.bool, device=target_raw.device).reshape(-1)
        if bool(earpiece_mask.any()):
            quantile = min(max(float(loss_cfg.get("earpiece_region_top_quantile", 0.95)), 0.0), 1.0)
            min_target = float(loss_cfg.get("earpiece_region_min_target", 0.0))
            earpiece_targets = target_raw.reshape(-1)[earpiece_mask]
            threshold = torch.quantile(earpiece_targets, quantile)
            peak_mask = earpiece_mask & (target_raw.reshape(-1) >= threshold) & (target_raw.reshape(-1) > min_target)
            weights = torch.where(peak_mask, weights * earpiece_weight, weights)
    return weights


def _compute_low_target_overprediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    weight = float(loss_cfg.get("low_target_overprediction_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())
    max_value = float(loss_cfg.get("low_target_overprediction_max", 100.0))
    min_value = float(loss_cfg.get("low_target_overprediction_min", 0.0))
    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    raw_flat = target_raw.reshape(-1)
    mask = (raw_flat >= min_value) & (raw_flat < max_value)
    if not bool(mask.any()):
        return prediction.new_zeros(())
    over_delta = (pred_flat[mask] - target_flat[mask]).clamp_min(0.0)
    return weight * F.smooth_l1_loss(over_delta, torch.zeros_like(over_delta), reduction="mean")


def _compute_censored_background_loss(
    prediction: torch.Tensor,
    target_raw: torch.Tensor,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
) -> torch.Tensor:
    weight = float(loss_cfg.get("censored_background_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())
    target_max = float(loss_cfg.get("censored_background_target_max", 100.0))
    pred_ceiling = float(loss_cfg.get("censored_background_pred_ceiling", target_max))
    margin_log = float(loss_cfg.get("censored_background_margin_log", 0.0))
    pred_log = _scaled_to_log(prediction, y_scaler).reshape(-1)
    target_raw_flat = target_raw.reshape(-1)
    mask = (target_raw_flat >= 0.0) & (target_raw_flat <= target_max)
    if not bool(mask.any()):
        return prediction.new_zeros(())
    ceiling_log = math.log1p(max(pred_ceiling, 0.0)) + margin_log
    over_delta = (pred_log[mask] - ceiling_log).clamp_min(0.0)
    if not bool((over_delta > 0.0).any()):
        return prediction.new_zeros(())
    return weight * F.smooth_l1_loss(over_delta, torch.zeros_like(over_delta), reduction="mean")


def _scaled_to_log(tensor: torch.Tensor, y_scaler: StandardScaler) -> torch.Tensor:
    mean = y_scaler.mean.to(tensor.device)
    std = y_scaler.std.to(tensor.device)
    return tensor * std + mean


def _case_group_tensor(case_names: list[str], device: torch.device) -> torch.Tensor:
    case_to_id: dict[str, int] = {}
    ids: list[int] = []
    for case_name in case_names:
        if case_name not in case_to_id:
            case_to_id[case_name] = len(case_to_id)
        ids.append(case_to_id[case_name])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _compute_background_false_peak_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
) -> torch.Tensor:
    weight = float(loss_cfg.get("background_false_peak_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())
    min_rank_fraction = float(loss_cfg.get("background_false_peak_min_rank_fraction", 0.50))
    margin_log = float(loss_cfg.get("background_false_peak_margin_log", math.log(1.25)))
    pred_log = _scaled_to_log(prediction, y_scaler).reshape(-1)
    target_log = _scaled_to_log(target, y_scaler).reshape(-1)
    target_raw = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)
    mask = torch.zeros_like(target_raw, dtype=torch.bool)
    quantile = min(max(1.0 - min_rank_fraction, 0.0), 1.0)
    sample_ids, counts = torch.unique_consecutive(sample_index, return_counts=True)
    if int(counts.sum().item()) == int(sample_index.numel()) and int(sample_ids.numel()) == int(
        torch.unique(sample_index).numel()
    ):
        start = 0
        for count in counts.tolist():
            end = start + int(count)
            sample_values = target_raw[start:end]
            if sample_values.numel() > 0:
                threshold = torch.quantile(sample_values, quantile)
                mask[start:end] = sample_values <= threshold
            start = end
    else:
        for sample_id in torch.unique(sample_index):
            sample_mask = sample_index == sample_id
            if not bool(sample_mask.any()):
                continue
            threshold = torch.quantile(target_raw[sample_mask], quantile)
            mask |= sample_mask & (target_raw <= threshold)
    if not bool(mask.any()):
        return prediction.new_zeros(())
    over_delta = (pred_log[mask] - target_log[mask] - margin_log).clamp_min(0.0)
    if not bool((over_delta > 0.0).any()):
        return prediction.new_zeros(())
    return weight * F.smooth_l1_loss(over_delta, torch.zeros_like(over_delta), reduction="mean")


def _loss_by_rank_bands(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    target_raw_flat: torch.Tensor,
    sample_index_flat: torch.Tensor,
    bands: list[dict[str, Any]],
    sample_quantiles: dict[float, torch.Tensor] | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    sample_ids = torch.unique(sample_index_flat)
    for band in bands:
        if not isinstance(band, dict):
            continue
        weight = float(band.get("weight", 1.0))
        if weight <= 0.0:
            continue
        left = float(band.get("left", band.get("min", 0.0)))
        right = float(band.get("right", band.get("max", 1.0)))
        lower_q = min(max(1.0 - right, 0.0), 1.0)
        upper_q = min(max(1.0 - left, 0.0), 1.0)
        band_losses: list[torch.Tensor] = []
        for sample_id in sample_ids:
            sample_mask = sample_index_flat == sample_id
            if not bool(sample_mask.any()):
                continue
            sample_values = target_raw_flat[sample_mask]
            if sample_values.numel() <= 0:
                continue
            lower = (
                sample_quantiles[lower_q][sample_id]
                if sample_quantiles is not None and lower_q in sample_quantiles
                else torch.quantile(sample_values, lower_q)
            )
            mask = sample_mask & (target_raw_flat >= lower)
            if left > 0.0:
                upper = (
                    sample_quantiles[upper_q][sample_id]
                    if sample_quantiles is not None and upper_q in sample_quantiles
                    else torch.quantile(sample_values, upper_q)
                )
                mask &= target_raw_flat < upper
            if bool(mask.any()):
                band_losses.append(F.smooth_l1_loss(pred_flat[mask], target_flat[mask], reduction="mean"))
        if band_losses:
            losses.append(weight * torch.stack(band_losses).mean())
    if not losses:
        return pred_flat.new_zeros(())
    return torch.stack(losses).mean()


def _compute_rank_band_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    weight = float(loss_cfg.get("rank_band_balanced_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())
    bands = loss_cfg.get("rank_band_balanced_bands")
    if not bands:
        bands = [
            {"left": 0.0, "right": 0.01, "weight": 1.0},
            {"left": 0.01, "right": 0.05, "weight": 1.0},
            {"left": 0.05, "right": 0.10, "weight": 1.0},
            {"left": 0.10, "right": 0.25, "weight": 1.0},
            {"left": 0.25, "right": 0.50, "weight": 1.0},
            {"left": 0.50, "right": 1.0, "weight": 1.0},
        ]
    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    target_raw_flat = target_raw.reshape(-1)
    sample_index_flat = sample_index.reshape(-1)
    return weight * _loss_by_rank_bands(pred_flat, target_flat, target_raw_flat, sample_index_flat, list(bands))


def _compute_participation_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
) -> torch.Tensor:
    weight = float(loss_cfg.get("participation_residual_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())

    pred_log = _scaled_to_log(prediction, y_scaler).reshape(-1)
    target_log = _scaled_to_log(target, y_scaler).reshape(-1)
    target_raw = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)

    baseline_mode = str(loss_cfg.get("participation_baseline", "mean_log")).lower()
    if baseline_mode == "mean_log":
        sample_count = int(sample_index.max().item()) + 1 if sample_index.numel() else 0
        if sample_count <= 0:
            return prediction.new_zeros(())
        counts = torch.bincount(sample_index, minlength=sample_count).to(pred_log.dtype).clamp_min(1.0)
        pred_sum = pred_log.new_zeros(sample_count).index_add_(0, sample_index, pred_log)
        target_sum = target_log.new_zeros(sample_count).index_add_(0, sample_index, target_log)
        pred_residual = pred_log - (pred_sum / counts)[sample_index]
        target_residual = target_log - (target_sum / counts)[sample_index]
        residual_loss = F.smooth_l1_loss(pred_residual, target_residual, reduction="mean")
        bands = loss_cfg.get("participation_residual_rank_bands")
        band_weight = float(loss_cfg.get("participation_residual_band_weight", 0.0))
        if band_weight > 0.0 and bands:
            quantiles = {
                min(max(1.0 - float(band.get(edge, band.get({"left": "min", "right": "max"}[edge], 0.0))), 0.0), 1.0)
                for band in bands
                if isinstance(band, dict)
                for edge in ("left", "right")
            }
            sample_quantiles: dict[float, torch.Tensor] = {}
            for quantile in quantiles:
                values = pred_log.new_empty(sample_count)
                for sample_id in range(sample_count):
                    sample_values = target_raw[sample_index == sample_id]
                    values[sample_id] = (
                        torch.quantile(sample_values, quantile)
                        if sample_values.numel() > 0
                        else target_raw.new_tensor(0.0)
                    )
                sample_quantiles[quantile] = values
            band_loss = _loss_by_rank_bands(
                pred_log,
                target_log,
                target_raw,
                sample_index,
                list(bands),
                sample_quantiles=sample_quantiles,
            )
            residual_loss = residual_loss + band_weight * band_loss
        return weight * residual_loss

    losses: list[torch.Tensor] = []
    for sample_id in torch.unique(sample_index):
        sample_mask = sample_index == sample_id
        if not bool(sample_mask.any()):
            continue
        sample_pred_log = pred_log[sample_mask]
        sample_target_log = target_log[sample_mask]
        sample_target_raw = target_raw[sample_mask]
        if baseline_mode == "top25_log_mean":
            threshold = torch.quantile(sample_target_raw, 0.75)
            baseline_mask = sample_target_raw >= threshold
            target_baseline = sample_target_log[baseline_mask].mean() if bool(baseline_mask.any()) else sample_target_log.mean()
            pred_baseline = sample_pred_log[baseline_mask].mean() if bool(baseline_mask.any()) else sample_pred_log.mean()
        elif baseline_mode == "top5_log_mean":
            threshold = torch.quantile(sample_target_raw, 0.95)
            baseline_mask = sample_target_raw >= threshold
            target_baseline = sample_target_log[baseline_mask].mean() if bool(baseline_mask.any()) else sample_target_log.mean()
            pred_baseline = sample_pred_log[baseline_mask].mean() if bool(baseline_mask.any()) else sample_pred_log.mean()
        elif baseline_mode == "logmean_raw":
            target_baseline = torch.log1p(sample_target_raw.mean().clamp_min(0.0))
            pred_baseline = torch.log1p(torch.expm1(sample_pred_log).clamp_min(0.0).mean())
        else:
            target_baseline = sample_target_log.mean()
            pred_baseline = sample_pred_log.mean()
        pred_residual = sample_pred_log - pred_baseline
        target_residual = sample_target_log - target_baseline
        losses.append(F.smooth_l1_loss(pred_residual, target_residual, reduction="mean"))

    if not losses:
        return prediction.new_zeros(())

    residual_loss = torch.stack(losses).mean()
    bands = loss_cfg.get("participation_residual_rank_bands")
    band_weight = float(loss_cfg.get("participation_residual_band_weight", 0.0))
    if band_weight > 0.0 and bands:
        band_loss = _loss_by_rank_bands(pred_log, target_log, target_raw, sample_index, list(bands))
        residual_loss = residual_loss + band_weight * band_loss
    return weight * residual_loss


def _compute_sample_tail_aux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    peak_weight = float(loss_cfg.get("sample_peak_loss_weight", 0.0))
    top5_weight = float(loss_cfg.get("sample_top5_loss_weight", 0.0))
    top1_weight = float(loss_cfg.get("sample_top1_loss_weight", 0.0))
    mean_weight = float(loss_cfg.get("sample_mean_loss_weight", 0.0))
    if peak_weight <= 0.0 and top5_weight <= 0.0 and top1_weight <= 0.0 and mean_weight <= 0.0:
        return prediction.new_zeros(())

    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    target_raw_flat = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)
    top5_quantile = float(loss_cfg.get("sample_top5_quantile", 0.95))
    top1_quantile = float(loss_cfg.get("sample_top1_quantile", 0.99))

    losses: list[torch.Tensor] = []
    for sample_id in torch.unique(sample_index):
        mask = sample_index == sample_id
        if not bool(mask.any()):
            continue
        sample_pred = pred_flat[mask]
        sample_target = target_flat[mask]
        sample_target_raw = target_raw_flat[mask]

        sample_losses: list[torch.Tensor] = []
        if peak_weight > 0.0:
            sample_losses.append(
                peak_weight
                * F.smooth_l1_loss(
                    sample_pred.max().reshape(1),
                    sample_target.max().reshape(1),
                    reduction="mean",
                )
            )
        if mean_weight > 0.0:
            sample_losses.append(
                mean_weight
                * F.smooth_l1_loss(
                    sample_pred.mean().reshape(1),
                    sample_target.mean().reshape(1),
                    reduction="mean",
                )
            )
        if top5_weight > 0.0 and sample_target_raw.numel() > 0:
            threshold = torch.quantile(sample_target_raw, min(max(top5_quantile, 0.0), 1.0))
            top_mask = sample_target_raw >= threshold
            if bool(top_mask.any()):
                sample_losses.append(
                    top5_weight
                    * F.smooth_l1_loss(sample_pred[top_mask], sample_target[top_mask], reduction="mean")
                )
        if top1_weight > 0.0 and sample_target_raw.numel() > 0:
            threshold = torch.quantile(sample_target_raw, min(max(top1_quantile, 0.0), 1.0))
            top_mask = sample_target_raw >= threshold
            if bool(top_mask.any()):
                sample_losses.append(
                    top1_weight
                    * F.smooth_l1_loss(sample_pred[top_mask], sample_target[top_mask], reduction="mean")
                )

        if sample_losses:
            losses.append(torch.stack(sample_losses).sum())

    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _compute_disk_center_tail_aux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    region_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    peak_weight = float(loss_cfg.get("disk_center_peak_loss_weight", 0.0))
    top5_weight = float(loss_cfg.get("disk_center_top5_loss_weight", 0.0))
    top1_weight = float(loss_cfg.get("disk_center_top1_loss_weight", 0.0))
    mean_weight = float(loss_cfg.get("disk_center_mean_loss_weight", 0.0))
    if peak_weight <= 0.0 and top5_weight <= 0.0 and top1_weight <= 0.0 and mean_weight <= 0.0:
        return prediction.new_zeros(())
    if region_mask is None:
        return prediction.new_zeros(())

    pred_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    target_raw_flat = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)
    region_mask = region_mask.to(dtype=torch.bool, device=prediction.device).reshape(-1)
    if not bool(region_mask.any()):
        return prediction.new_zeros(())

    top5_quantile = float(loss_cfg.get("disk_center_top5_quantile", loss_cfg.get("sample_top5_quantile", 0.95)))
    top1_quantile = float(loss_cfg.get("disk_center_top1_quantile", loss_cfg.get("sample_top1_quantile", 0.99)))
    min_points = max(1, int(loss_cfg.get("disk_center_min_points", 1)))
    min_target = float(loss_cfg.get("disk_center_min_target", 0.0))

    losses: list[torch.Tensor] = []
    for sample_id in torch.unique(sample_index):
        mask = (sample_index == sample_id) & region_mask
        if int(mask.sum().item()) < min_points:
            continue
        sample_pred = pred_flat[mask]
        sample_target = target_flat[mask]
        sample_target_raw = target_raw_flat[mask]

        sample_losses: list[torch.Tensor] = []
        if peak_weight > 0.0:
            sample_losses.append(
                peak_weight
                * F.smooth_l1_loss(
                    sample_pred.max().reshape(1),
                    sample_target.max().reshape(1),
                    reduction="mean",
                )
            )
        if mean_weight > 0.0:
            sample_losses.append(
                mean_weight
                * F.smooth_l1_loss(
                    sample_pred.mean().reshape(1),
                    sample_target.mean().reshape(1),
                    reduction="mean",
                )
            )
        if top5_weight > 0.0 and sample_target_raw.numel() > 0:
            threshold = torch.quantile(sample_target_raw, min(max(top5_quantile, 0.0), 1.0))
            top_mask = (sample_target_raw >= threshold) & (sample_target_raw > min_target)
            if bool(top_mask.any()):
                sample_losses.append(
                    top5_weight
                    * F.smooth_l1_loss(sample_pred[top_mask], sample_target[top_mask], reduction="mean")
                )
        if top1_weight > 0.0 and sample_target_raw.numel() > 0:
            threshold = torch.quantile(sample_target_raw, min(max(top1_quantile, 0.0), 1.0))
            top_mask = (sample_target_raw >= threshold) & (sample_target_raw > min_target)
            if bool(top_mask.any()):
                sample_losses.append(
                    top1_weight
                    * F.smooth_l1_loss(sample_pred[top_mask], sample_target[top_mask], reduction="mean")
                )

        if sample_losses:
            losses.append(torch.stack(sample_losses).sum())

    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _compute_region_tail_aux_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    region_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
    prefix: str,
) -> torch.Tensor:
    if prefix == "disk_center":
        return _compute_disk_center_tail_aux_loss(
            prediction=prediction,
            target=target,
            target_raw=target_raw,
            sample_index=sample_index,
            region_mask=region_mask,
            loss_cfg=loss_cfg,
        )

    region_cfg = dict(loss_cfg)
    region_cfg["disk_center_peak_loss_weight"] = loss_cfg.get(f"{prefix}_peak_loss_weight", 0.0)
    region_cfg["disk_center_top5_loss_weight"] = loss_cfg.get(f"{prefix}_top5_loss_weight", 0.0)
    region_cfg["disk_center_top1_loss_weight"] = loss_cfg.get(f"{prefix}_top1_loss_weight", 0.0)
    region_cfg["disk_center_mean_loss_weight"] = loss_cfg.get(f"{prefix}_mean_loss_weight", 0.0)
    region_cfg["disk_center_top5_quantile"] = loss_cfg.get(
        f"{prefix}_top5_quantile",
        loss_cfg.get("sample_top5_quantile", 0.95),
    )
    region_cfg["disk_center_top1_quantile"] = loss_cfg.get(
        f"{prefix}_top1_quantile",
        loss_cfg.get("sample_top1_quantile", 0.99),
    )
    region_cfg["disk_center_min_points"] = loss_cfg.get(f"{prefix}_min_points", 1)
    region_cfg["disk_center_min_target"] = loss_cfg.get(f"{prefix}_min_target", 0.0)
    return _compute_disk_center_tail_aux_loss(
        prediction=prediction,
        target=target,
        target_raw=target_raw,
        sample_index=sample_index,
        region_mask=region_mask,
        loss_cfg=region_cfg,
    )


def _compute_disk_center_underprediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    region_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
) -> torch.Tensor:
    top5_weight = float(loss_cfg.get("disk_center_top5_under_weight", 0.0))
    top1_weight = float(loss_cfg.get("disk_center_top1_under_weight", 0.0))
    if top5_weight <= 0.0 and top1_weight <= 0.0:
        return prediction.new_zeros(())
    if region_mask is None:
        return prediction.new_zeros(())

    pred_log = _scaled_to_log(prediction, y_scaler).reshape(-1)
    target_log = _scaled_to_log(target, y_scaler).reshape(-1)
    target_raw_flat = target_raw.reshape(-1)
    sample_index = sample_index.reshape(-1)
    region_mask = region_mask.to(dtype=torch.bool, device=prediction.device).reshape(-1)
    if not bool(region_mask.any()):
        return prediction.new_zeros(())

    top5_quantile = float(
        loss_cfg.get(
            "disk_center_under_top5_quantile",
            loss_cfg.get("disk_center_top5_quantile", loss_cfg.get("sample_top5_quantile", 0.95)),
        )
    )
    top1_quantile = float(
        loss_cfg.get(
            "disk_center_under_top1_quantile",
            loss_cfg.get("disk_center_top1_quantile", loss_cfg.get("sample_top1_quantile", 0.99)),
        )
    )
    min_points = max(1, int(loss_cfg.get("disk_center_min_points", 1)))
    min_target = float(loss_cfg.get("disk_center_min_target", 0.0))
    margin_log = float(loss_cfg.get("disk_center_under_margin_log", math.log(1.10)))

    losses: list[torch.Tensor] = []
    for sample_id in torch.unique(sample_index):
        mask = (sample_index == sample_id) & region_mask
        if int(mask.sum().item()) < min_points:
            continue
        sample_target_raw = target_raw_flat[mask]
        sample_pred_log = pred_log[mask]
        sample_target_log = target_log[mask]

        sample_losses: list[torch.Tensor] = []
        if top5_weight > 0.0:
            threshold = torch.quantile(sample_target_raw, min(max(top5_quantile, 0.0), 1.0))
            top_mask = (sample_target_raw >= threshold) & (sample_target_raw > min_target)
            if bool(top_mask.any()):
                under_delta = (sample_target_log[top_mask] - sample_pred_log[top_mask] - margin_log).clamp_min(0.0)
                if bool((under_delta > 0.0).any()):
                    sample_losses.append(
                        top5_weight
                        * F.smooth_l1_loss(under_delta, torch.zeros_like(under_delta), reduction="mean")
                    )
        if top1_weight > 0.0:
            threshold = torch.quantile(sample_target_raw, min(max(top1_quantile, 0.0), 1.0))
            top_mask = (sample_target_raw >= threshold) & (sample_target_raw > min_target)
            if bool(top_mask.any()):
                under_delta = (sample_target_log[top_mask] - sample_pred_log[top_mask] - margin_log).clamp_min(0.0)
                if bool((under_delta > 0.0).any()):
                    sample_losses.append(
                        top1_weight
                        * F.smooth_l1_loss(under_delta, torch.zeros_like(under_delta), reduction="mean")
                    )

        if sample_losses:
            losses.append(torch.stack(sample_losses).sum())

    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _compute_region_underprediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sample_index: torch.Tensor,
    region_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
    prefix: str,
) -> torch.Tensor:
    if prefix == "disk_center":
        return _compute_disk_center_underprediction_loss(
            prediction=prediction,
            target=target,
            target_raw=target_raw,
            sample_index=sample_index,
            region_mask=region_mask,
            loss_cfg=loss_cfg,
            y_scaler=y_scaler,
        )

    region_cfg = dict(loss_cfg)
    region_cfg["disk_center_top5_under_weight"] = loss_cfg.get(f"{prefix}_top5_under_weight", 0.0)
    region_cfg["disk_center_top1_under_weight"] = loss_cfg.get(f"{prefix}_top1_under_weight", 0.0)
    region_cfg["disk_center_under_top5_quantile"] = loss_cfg.get(
        f"{prefix}_under_top5_quantile",
        loss_cfg.get(f"{prefix}_top5_quantile", loss_cfg.get("sample_top5_quantile", 0.95)),
    )
    region_cfg["disk_center_under_top1_quantile"] = loss_cfg.get(
        f"{prefix}_under_top1_quantile",
        loss_cfg.get(f"{prefix}_top1_quantile", loss_cfg.get("sample_top1_quantile", 0.99)),
    )
    region_cfg["disk_center_min_points"] = loss_cfg.get(f"{prefix}_min_points", 1)
    region_cfg["disk_center_min_target"] = loss_cfg.get(f"{prefix}_min_target", 0.0)
    region_cfg["disk_center_under_margin_log"] = loss_cfg.get(
        f"{prefix}_under_margin_log",
        math.log(1.10),
    )
    return _compute_disk_center_underprediction_loss(
        prediction=prediction,
        target=target,
        target_raw=target_raw,
        sample_index=sample_index,
        region_mask=region_mask,
        loss_cfg=region_cfg,
        y_scaler=y_scaler,
    )


def _compute_center_curve_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_index: torch.Tensor,
    node_indices: torch.Tensor,
    case_group: torch.Tensor,
    region_mask: torch.Tensor | None,
    loss_cfg: dict[str, Any],
    y_scaler: StandardScaler,
) -> torch.Tensor:
    weight = float(loss_cfg.get("center_curve_consistency_weight", 0.0))
    if weight <= 0.0:
        return prediction.new_zeros(())
    if region_mask is None:
        return prediction.new_zeros(())

    min_points = max(2, int(loss_cfg.get("center_curve_consistency_min_points", 8)))
    min_frequencies = max(2, int(loss_cfg.get("center_curve_consistency_min_frequencies", 3)))
    top_fraction = float(loss_cfg.get("center_curve_consistency_top_fraction", 0.05))
    pred_log = _scaled_to_log(prediction, y_scaler).reshape(-1)
    target_log = _scaled_to_log(target, y_scaler).reshape(-1)
    sample_index = sample_index.reshape(-1)
    node_indices = node_indices.reshape(-1)
    region_mask = region_mask.to(dtype=torch.bool, device=prediction.device).reshape(-1)
    point_case_group = case_group[sample_index].reshape(-1)
    max_nodes = max(1, int(loss_cfg.get("center_curve_consistency_max_nodes", 128)))
    losses: list[torch.Tensor] = []

    for case_id in torch.unique(point_case_group):
        case_mask = (point_case_group == case_id) & region_mask
        if not bool(case_mask.any()):
            continue
        sample_ids = torch.unique(sample_index[case_mask])
        if int(sample_ids.numel()) < min_frequencies:
            continue
        top_count = max(min_points, int(math.ceil(int(case_mask.sum().item()) * top_fraction)))
        top_count = min(top_count, int(case_mask.sum().item()))
        top_positions = torch.nonzero(case_mask, as_tuple=False).reshape(-1)[
            torch.topk(target_log[case_mask], k=top_count, largest=True).indices
        ]
        candidate_nodes = torch.unique(node_indices[top_positions])
        if int(candidate_nodes.numel()) > max_nodes:
            node_scores = []
            for node_id in candidate_nodes:
                node_mask = case_mask & (node_indices == node_id)
                node_scores.append(target_log[node_mask].max())
            score_tensor = torch.stack(node_scores)
            candidate_nodes = candidate_nodes[torch.topk(score_tensor, k=max_nodes, largest=True).indices]

        node_curve_losses: list[torch.Tensor] = []
        for node_id in candidate_nodes:
            node_mask = case_mask & (node_indices == node_id)
            if int(node_mask.sum().item()) < min_frequencies:
                continue
            order = torch.argsort(sample_index[node_mask])
            node_positions = torch.nonzero(node_mask, as_tuple=False).reshape(-1)[order]
            if int(node_positions.numel()) < min_frequencies:
                continue
            node_curve_losses.append(
                F.smooth_l1_loss(
                    pred_log[node_positions] - pred_log[node_positions].mean(),
                    target_log[node_positions] - target_log[node_positions].mean(),
                    reduction="mean",
                )
            )
        if node_curve_losses:
            losses.append(torch.stack(node_curve_losses).mean())

    if not losses:
        return prediction.new_zeros(())
    return weight * torch.stack(losses).mean()


def prepare_point_sample(
    raw: RawPointSample,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any] | None = None,
    target_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
) -> PreparedPointSample:
    scaled_features = x_scaler.transform(raw.scaled_features)
    features = torch.cat([raw.geometry_features, scaled_features, raw.mask_features], dim=-1)
    schema = feature_schema or {}
    target_config = target_cfg or {}
    target_floor = _resolve_threshold_from_config(target_config, schema, "zero_below") if schema else 0.0
    target_raw = _apply_target_floor(raw.target_raw, target_floor)
    target_log = torch.log1p(target_raw).unsqueeze(-1)
    target_scaled = y_scaler.transform(target_log)
    region_masks = None
    if raw.region_masks:
        region_masks = {name: mask.to(dtype=torch.bool) for name, mask in raw.region_masks.items()}
    point_weights = _build_point_weights(
        target_raw,
        schema,
        loss_cfg or {},
        region_masks=region_masks,
    )
    return PreparedPointSample(
        name=raw.name,
        case_name=raw.case_name,
        frequency_hz=raw.frequency_hz,
        features=features.to(dtype=torch.float32),
        target_scaled=target_scaled.to(dtype=torch.float32),
        target_log=target_log.to(dtype=torch.float32),
        target_raw=target_raw.to(dtype=torch.float32),
        node_indices=raw.node_indices,
        point_weights=point_weights.to(dtype=torch.float32),
        region_masks=region_masks,
    )


class PointSampleDataset(Dataset[PreparedPointSample]):
    def __init__(
        self,
        sample_paths: list[Path],
        dataset_cfg: dict[str, Any],
        feature_cfg: dict[str, Any],
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        feature_schema: dict[str, Any],
        target_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        cache_prepared_samples: bool = False,
        max_cached_samples: int | None = None,
    ) -> None:
        self.sample_paths = list(sample_paths)
        self.dataset_cfg = dict(dataset_cfg)
        self.feature_cfg = dict(feature_cfg)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_schema = dict(feature_schema)
        self.target_cfg = dict(target_cfg)
        self.loss_cfg = dict(loss_cfg)
        self.cache_prepared_samples = bool(cache_prepared_samples)
        self.max_cached_samples = None if max_cached_samples is None else max(1, int(max_cached_samples))
        self._cache: OrderedDict[int, PreparedPointSample] = OrderedDict()

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> PreparedPointSample:
        if self.cache_prepared_samples and index in self._cache:
            cached = self._cache.pop(index)
            self._cache[index] = cached
            return cached

        raw = load_raw_point_sample(
            self.sample_paths[index],
            dataset_cfg=self.dataset_cfg,
            feature_cfg=self.feature_cfg,
        )
        prepared = prepare_point_sample(
            raw,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            feature_schema=self.feature_schema,
            target_cfg=self.target_cfg,
            loss_cfg=self.loss_cfg,
        )
        if self.cache_prepared_samples:
            self._cache[index] = prepared
            if self.max_cached_samples is not None:
                while len(self._cache) > self.max_cached_samples:
                    self._cache.popitem(last=False)
        return prepared


def collate_point_samples(samples: list[PreparedPointSample]) -> PointBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    features = []
    target_scaled = []
    target_log = []
    target_raw = []
    point_weights = []
    sample_indices = []
    node_indices = []
    names = []
    case_names = []
    frequencies = []
    region_mask_names = sorted(
        {
            name
            for sample in samples
            for name in ((sample.region_masks or {}).keys())
        }
    )
    region_masks: dict[str, list[torch.Tensor]] = {name: [] for name in region_mask_names}
    for sample_idx, sample in enumerate(samples):
        names.append(sample.name)
        case_names.append(sample.case_name)
        frequencies.append(float(sample.frequency_hz))
        features.append(sample.features)
        target_scaled.append(sample.target_scaled)
        target_log.append(sample.target_log)
        target_raw.append(sample.target_raw)
        point_weights.append(sample.point_weights)
        node_indices.append(sample.node_indices)
        sample_indices.append(torch.full((sample.num_points,), sample_idx, dtype=torch.long))
        for name in region_mask_names:
            mask = (sample.region_masks or {}).get(name)
            if mask is None:
                mask = torch.zeros((sample.num_points,), dtype=torch.bool)
            region_masks[name].append(mask.to(dtype=torch.bool).reshape(-1))

    return PointBatch(
        names=names,
        case_names=case_names,
        frequency_hz=torch.tensor(frequencies, dtype=torch.float32),
        features=torch.cat(features, dim=0),
        target_scaled=torch.cat(target_scaled, dim=0),
        target_log=torch.cat(target_log, dim=0),
        target_raw=torch.cat(target_raw, dim=0),
        sample_index=torch.cat(sample_indices, dim=0),
        node_indices=torch.cat(node_indices, dim=0),
        point_weights=torch.cat(point_weights, dim=0),
        region_masks={name: torch.cat(parts, dim=0) for name, parts in region_masks.items()} if region_masks else None,
    )


def _load_scaler_sample(
    sample_path: Path,
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    target_zero_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, RawPointSample]:
    raw = load_raw_point_sample(sample_path, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
    target_raw = _apply_target_floor(raw.target_raw, target_zero_threshold)
    return raw.scaled_features, torch.log1p(target_raw).unsqueeze(-1), raw


def _load_target_values_for_stats(
    sample_path: Path,
    dataset_cfg: dict[str, Any],
) -> np.ndarray:
    case_dir = sample_path.parent.parent
    nodes_header = pd.read_csv(case_dir / "nodes.csv", nrows=0).columns
    nodes_usecols = ["node_index"] if "node_index" in nodes_header else None
    nodes_df = pd.read_csv(case_dir / "nodes.csv", usecols=nodes_usecols)
    target_df = pd.read_csv(
        sample_path,
        usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN},
    )
    target_values = _load_aligned_target_column(
        target_df,
        target_column=PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_df,
        target_path=sample_path,
    )
    selected_mask = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg, target_values=target_values)
    values = target_values[selected_mask]
    values = values[np.isfinite(values)]
    values = values[values >= 0.0]
    return values.astype(np.float64, copy=False)


def estimate_target_quantiles(
    train_sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    sample_limit: int | None,
    sample_seed: int,
    num_workers: int,
    max_values: int | None = None,
    logger: Any | None = None,
) -> dict[str, float]:
    selected_paths = list(train_sample_paths)
    if sample_limit is not None:
        limit = int(sample_limit)
        if limit < len(selected_paths):
            selected_paths = random.Random(int(sample_seed)).sample(selected_paths, limit)
        else:
            selected_paths = selected_paths[:limit]
    if not selected_paths:
        return {}

    started_at = time.monotonic()
    workers = max(1, int(num_workers))
    values: list[np.ndarray] = []
    total_points = 0
    positive_points = 0
    per_file_cap = None
    if max_values is not None and int(max_values) > 0:
        per_file_cap = max(1, int(max_values) // len(selected_paths))
    if logger is not None:
        logger.info("Estimating target quantiles | samples=%s | workers=%s", len(selected_paths), workers)

    def consume(path: Path, array: np.ndarray) -> None:
        nonlocal total_points, positive_points
        total_points += int(array.size)
        positive_points += int((array > 0.0).sum())
        if per_file_cap is not None and array.size > per_file_cap:
            seed = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
            indices = np.random.default_rng(seed).choice(array.size, size=per_file_cap, replace=False)
            array = array[indices]
        values.append(array)

    if workers == 1:
        for idx, path in enumerate(selected_paths, start=1):
            consume(path, _load_target_values_for_stats(path, dataset_cfg))
            if logger is not None and (idx == 1 or idx == len(selected_paths) or idx % 100 == 0):
                logger.info(
                    "Target quantile progress | sample=%s/%s | points=%s | elapsed=%s",
                    idx,
                    len(selected_paths),
                    total_points,
                    _format_duration(time.monotonic() - started_at),
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            path_iter = iter(selected_paths)
            max_pending = max(workers, workers * 4)
            pending: dict[Any, Path] = {}

            def submit_until_full() -> None:
                while len(pending) < max_pending:
                    try:
                        path = next(path_iter)
                    except StopIteration:
                        return
                    pending[executor.submit(_load_target_values_for_stats, path, dataset_cfg)] = path

            submit_until_full()
            idx = 0
            while pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    path = pending.pop(future)
                    idx += 1
                    try:
                        consume(path, future.result())
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load target stats sample: {path}") from exc
                if logger is not None and (idx == 1 or idx == len(selected_paths) or idx % 100 == 0):
                    logger.info(
                        "Target quantile progress | sample=%s/%s | points=%s | elapsed=%s",
                        idx,
                        len(selected_paths),
                        total_points,
                        _format_duration(time.monotonic() - started_at),
                    )
                submit_until_full()

    concatenated = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    positive = concatenated[concatenated > 0.0]
    if positive.size == 0:
        return {"target_points_for_quantiles": float(total_points), "target_positive_points_for_quantiles": 0.0}

    return {
        "target_points_for_quantiles": float(total_points),
        "target_positive_points_for_quantiles": float(positive_points),
        "target_positive_p50": float(np.quantile(positive, 0.50)),
        "target_positive_p90": float(np.quantile(positive, 0.90)),
        "target_positive_p95": float(np.quantile(positive, 0.95)),
        "target_positive_p99": float(np.quantile(positive, 0.99)),
        "target_positive_p999": float(np.quantile(positive, 0.999)),
    }


def fit_scalers(
    train_sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    num_workers: int,
    sample_limit: int | None = None,
    sample_seed: int = 42,
    prefetch_factor: int = 4,
    target_zero_threshold: float = 0.0,
    target_stats: dict[str, float] | None = None,
    logger: Any | None = None,
) -> tuple[StandardScaler, StandardScaler, dict[str, Any]]:
    selected_paths = list(train_sample_paths)
    if sample_limit is not None:
        sample_limit_int = int(sample_limit)
        if sample_limit_int < len(selected_paths):
            rng = random.Random(int(sample_seed))
            selected_paths = rng.sample(selected_paths, sample_limit_int)
        else:
            selected_paths = selected_paths[:sample_limit_int]
    if not selected_paths:
        raise ValueError("Training split has no samples; cannot fit scalers.")

    x_stats = RunningTensorStats()
    y_stats = RunningTensorStats()
    feature_schema: dict[str, Any] | None = None
    total_points = 0
    started_at = time.monotonic()
    workers = max(1, int(num_workers))
    max_pending = max(workers, workers * max(1, int(prefetch_factor)))

    if logger is not None:
        logger.info(
            "Fitting scalers | samples=%s | workers=%s | max_pending=%s | target_zero_threshold=%.6g",
            len(selected_paths),
            workers,
            max_pending,
            float(target_zero_threshold),
        )

    if workers == 1:
        iterator = (
            _load_scaler_sample(
                path,
                dataset_cfg=dataset_cfg,
                feature_cfg=feature_cfg,
                target_zero_threshold=target_zero_threshold,
            )
            for path in selected_paths
        )
        for idx, (x_values, y_values, raw) in enumerate(iterator, start=1):
            if raw.num_points <= 0:
                continue
            x_stats.update(x_values)
            y_stats.update(y_values)
            total_points += raw.num_points
            if feature_schema is None:
                feature_schema = _schema_from_raw_sample(raw)
            if logger is not None and (idx == 1 or idx == len(selected_paths) or idx % 100 == 0):
                logger.info(
                    "Scaler progress | sample=%s/%s | points=%s | elapsed=%s",
                    idx,
                    len(selected_paths),
                    total_points,
                    _format_duration(time.monotonic() - started_at),
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            path_iter = iter(selected_paths)
            pending: dict[Any, Path] = {}

            def submit_until_full() -> None:
                while len(pending) < max_pending:
                    try:
                        path = next(path_iter)
                    except StopIteration:
                        return
                    pending[executor.submit(_load_scaler_sample, path, dataset_cfg, feature_cfg, target_zero_threshold)] = path

            submit_until_full()
            completed = 0
            while pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    path = pending.pop(future)
                    completed += 1
                    try:
                        x_values, y_values, raw = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load scaler sample: {path}") from exc
                    if raw.num_points <= 0:
                        continue
                    x_stats.update(x_values)
                    y_stats.update(y_values)
                    total_points += raw.num_points
                    if feature_schema is None:
                        feature_schema = _schema_from_raw_sample(raw)
                    if logger is not None and (
                        completed == 1 or completed == len(selected_paths) or completed % 100 == 0
                    ):
                        logger.info(
                            "Scaler progress | sample=%s/%s | points=%s | elapsed=%s",
                            completed,
                            len(selected_paths),
                            total_points,
                            _format_duration(time.monotonic() - started_at),
                        )
                submit_until_full()
                if logger is not None and completed > 0 and completed % 1000 == 0:
                    logger.info(
                        "Scaler queue | completed=%s/%s | pending=%s | elapsed=%s",
                        completed,
                        len(selected_paths),
                        len(pending),
                        _format_duration(time.monotonic() - started_at),
                    )

    if feature_schema is None:
        raise ValueError("No valid non-negative earpiece points were found in the scaler fit set.")
    feature_schema["scaler_fit_points"] = total_points
    if target_stats:
        feature_schema.update(target_stats)
    feature_schema["target_zero_threshold"] = float(target_zero_threshold)
    return x_stats.finalize(), y_stats.finalize(), feature_schema


def _schema_from_raw_sample(raw: RawPointSample) -> dict[str, Any]:
    feature_names = raw.geometry_feature_names + raw.scaled_feature_names + raw.mask_feature_names
    return {
        "feature_names": feature_names,
        "geometry_normalized_feature_names": list(raw.geometry_feature_names),
        "scaled_continuous_feature_names": list(raw.scaled_feature_names),
        "mask_feature_names": list(raw.mask_feature_names),
        "input_dim": len(feature_names),
    }


def _probe_current_feature_schema(
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
) -> dict[str, Any]:
    for sample_path in sample_paths[: max(1, min(len(sample_paths), 16))]:
        raw = load_raw_point_sample(sample_path, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)
        if raw.num_points > 0:
            return _schema_from_raw_sample(raw)
    raise ValueError("Could not probe feature schema; sampled training files had no valid points.")


def _scaler_cache_matches_schema(
    x_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    current_schema: dict[str, Any],
    target_zero_threshold: float = 0.0,
) -> bool:
    if feature_schema.get("feature_names") != current_schema.get("feature_names"):
        return False
    cached_target_zero = float(feature_schema.get("target_zero_threshold", 0.0) or 0.0)
    if not math.isclose(cached_target_zero, float(target_zero_threshold), rel_tol=1e-9, abs_tol=1e-12):
        return False
    expected_scaled_dim = len(current_schema.get("scaled_continuous_feature_names", []))
    return int(x_scaler.mean.numel()) == expected_scaled_dim


def make_loader(
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    sample_batch_size: int,
    num_workers: int,
    shuffle: bool,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    pin_memory: bool | None = None,
    cache_prepared_samples: bool = False,
    max_cached_samples_per_worker: int | None = None,
) -> DataLoader[PointBatch]:
    dataset = PointSampleDataset(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        cache_prepared_samples=cache_prepared_samples,
        max_cached_samples=max_cached_samples_per_worker,
    )
    workers = max(0, int(num_workers))
    should_pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    return DataLoader(
        dataset,
        batch_size=max(1, int(sample_batch_size)),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_point_samples,
        pin_memory=should_pin_memory,
        persistent_workers=workers > 0 and bool(persistent_workers),
        prefetch_factor=max(1, int(prefetch_factor)) if workers > 0 and prefetch_factor is not None else (2 if workers > 0 else None),
    )


def _feature_indices_matching(
    feature_names: list[str],
    *,
    include_patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> list[int]:
    excludes = [str(item) for item in (exclude_patterns or [])]
    includes = [str(item) for item in include_patterns]
    indices: list[int] = []
    for index, name in enumerate(feature_names):
        if includes and not any(pattern in name for pattern in includes):
            continue
        if excludes and any(pattern in name for pattern in excludes):
            continue
        indices.append(index)
    return indices


def _low_rank_feature_indices(model_cfg: dict[str, Any], feature_schema: dict[str, Any]) -> tuple[list[int], list[int]]:
    feature_names = [str(name) for name in feature_schema.get("feature_names", [])]
    if not feature_names:
        return [], []
    if model_cfg.get("curve_frequency_feature_indices") is not None:
        frequency_indices = [int(index) for index in model_cfg.get("curve_frequency_feature_indices", [])]
    else:
        frequency_patterns = [
            str(pattern)
            for pattern in model_cfg.get(
                "curve_frequency_feature_patterns",
                [
                    "earpiece_thickness",
                    "earpiece_RadialDist",
                    "earpiece_TopWidth",
                    "earpiece_HoleTopDist",
                    "earpiece_TopFilletRadius",
                    "earpiece_BottomFilletRadius",
                    "plate_radius",
                    "Add_mass",
                    "plate_thickness",
                    "earpiece_HoleRadius",
                    "mass_couple_radius",
                    "psd_",
                    "psd_value_at_frequency",
                    "log_psd_value_at_frequency",
                    "frequency",
                    "log_frequency",
                    "freq_top",
                    "signed_delta_to_mode",
                    "abs_delta_to_mode",
                    "nearest_delta",
                    "first_mode_ratio",
                    "freq_ratio_mode_",
                    "modal_detuning_mode_",
                    "log_modal_amp_",
                    "modal_weight_",
                    "nearest_log_modal_amp_",
                    "sum_log_modal_amp_",
                    "top3_log_modal_amp_sum_",
                    "nearest_modal_weight_",
                    "log_modal_gain_frf",
                    "modal_weight_frf",
                    "freq_ratio_frf",
                ],
            )
        ]
        frequency_exclude_patterns = [
            str(pattern) for pattern in model_cfg.get("curve_frequency_feature_exclude_patterns", [])
        ]
        frequency_indices = _feature_indices_matching(
            feature_names,
            include_patterns=frequency_patterns,
            exclude_patterns=frequency_exclude_patterns,
        )

    if model_cfg.get("curve_node_feature_indices") is not None:
        node_indices = [int(index) for index in model_cfg.get("curve_node_feature_indices", [])]
    else:
        node_patterns = [
            str(pattern)
            for pattern in model_cfg.get(
                "curve_node_feature_patterns",
                [
                    "_norm",
                    "dist_to_",
                    "sin_theta",
                    "cos_theta",
                    "center_",
                    "near_",
                    "weighted_abs_",
                    "weighted_umag",
                    "nearest_abs_",
                    "nearest_umag",
                    "weighted_grad_",
                    "nearest_grad_",
                    "active1_abs_",
                    "active2_abs_",
                    "active3_abs_",
                    "active1_umag",
                    "active2_umag",
                    "active3_umag",
                    "active1_grad_",
                    "active2_grad_",
                    "active3_grad_",
                    "modal_baseline_log_",
                    "shape_rms",
                    "mask",
                    "stress_region_mask",
                ],
            )
        ]
        node_exclude_patterns = [str(pattern) for pattern in model_cfg.get("curve_node_feature_exclude_patterns", [])]
        node_indices = _feature_indices_matching(
            feature_names,
            include_patterns=node_patterns,
            exclude_patterns=node_exclude_patterns,
        )

    if not frequency_indices:
        frequency_indices = list(range(len(feature_names)))
    if not node_indices:
        node_indices = list(range(len(feature_names)))
    return frequency_indices, node_indices


def build_model(config: dict[str, Any], input_dim: int, feature_schema: dict[str, Any] | None = None) -> PointMLP:
    model_cfg = dict(config.get("model", {}))
    low_rank_enabled = bool(model_cfg.get("low_rank_curve_head", False))
    frequency_indices: list[int] = []
    node_indices: list[int] = []
    if low_rank_enabled:
        frequency_indices, node_indices = _low_rank_feature_indices(model_cfg, feature_schema or {})
    return PointMLP(
        input_dim=input_dim,
        hidden_dims=[int(dim) for dim in model_cfg.get("hidden_dims", [256, 256, 128])],
        dropout=float(model_cfg.get("dropout", 0.1)),
        activation=str(model_cfg.get("activation", "silu")),
        use_layer_norm=bool(model_cfg.get("layer_norm", True)),
        low_rank_curve_head=low_rank_enabled,
        curve_rank=int(model_cfg.get("center_curve_rank", model_cfg.get("curve_rank", 3))),
        frequency_feature_indices=frequency_indices,
        node_feature_indices=node_indices,
        residual_weight=float(model_cfg.get("center_residual_weight", model_cfg.get("residual_weight", 0.1))),
    )


def _load_checkpoint_for_feature_schema_change(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    current_feature_schema: dict[str, Any],
) -> dict[str, int]:
    checkpoint_state = checkpoint["model_state"]
    checkpoint_schema = dict(checkpoint.get("feature_schema", {}))
    old_feature_names = list(checkpoint_schema.get("feature_names", []))
    current_feature_names = list(current_feature_schema.get("feature_names", []))
    current_state = model.state_dict()
    adapted_state = {name: tensor.clone() for name, tensor in current_state.items()}
    copied = 0
    skipped = 0
    partial_first_layer = 0

    old_index = {name: idx for idx, name in enumerate(old_feature_names)}
    current_index = {name: idx for idx, name in enumerate(current_feature_names)}

    for name, tensor in checkpoint_state.items():
        if name not in adapted_state:
            skipped += 1
            continue
        if adapted_state[name].shape == tensor.shape:
            adapted_state[name].copy_(tensor)
            copied += 1
            continue
        if name == "network.0.weight" and name in adapted_state and tensor.dim() == 2 and adapted_state[name].dim() == 2:
            shared_names = [feature for feature in current_feature_names if feature in old_index]
            if shared_names:
                for feature in shared_names:
                    adapted_state[name][:, current_index[feature]].copy_(tensor[:, old_index[feature]])
                partial_first_layer = len(shared_names)
                copied += 1
            else:
                skipped += 1
            continue
        skipped += 1

    model.load_state_dict(adapted_state)
    return {
        "copied_tensors": copied,
        "skipped_tensors": skipped,
        "partial_first_layer_features": partial_first_layer,
        "old_input_dim": len(old_feature_names),
        "current_input_dim": len(current_feature_names),
    }


def _blend_first_layer_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    current_feature_schema: dict[str, Any],
    *,
    blend: float,
    feature_patterns: list[str] | None = None,
) -> dict[str, int | float]:
    checkpoint_state = checkpoint["model_state"]
    checkpoint_schema = dict(checkpoint.get("feature_schema", {}))
    old_feature_names = list(checkpoint_schema.get("feature_names", []))
    current_feature_names = list(current_feature_schema.get("feature_names", []))
    old_weight = checkpoint_state.get("network.0.weight")
    if old_weight is None:
        return {
            "blended_features": 0,
            "skipped_features": 0,
            "old_input_dim": len(old_feature_names),
            "current_input_dim": len(current_feature_names),
            "blend": float(blend),
        }

    current_state = model.state_dict()
    current_weight = current_state.get("network.0.weight")
    if current_weight is None or current_weight.dim() != 2 or old_weight.dim() != 2:
        return {
            "blended_features": 0,
            "skipped_features": len(current_feature_names),
            "old_input_dim": len(old_feature_names),
            "current_input_dim": len(current_feature_names),
            "blend": float(blend),
        }
    if int(current_weight.shape[0]) != int(old_weight.shape[0]):
        return {
            "blended_features": 0,
            "skipped_features": len(current_feature_names),
            "old_input_dim": len(old_feature_names),
            "current_input_dim": len(current_feature_names),
            "blend": float(blend),
        }

    patterns = [str(pattern) for pattern in (feature_patterns or []) if str(pattern)]
    old_index = {name: idx for idx, name in enumerate(old_feature_names)}
    current_index = {name: idx for idx, name in enumerate(current_feature_names)}
    alpha = min(max(float(blend), 0.0), 1.0)
    source_weight = old_weight.to(device=current_weight.device, dtype=current_weight.dtype)
    blended = 0
    skipped = 0
    with torch.no_grad():
        for feature_name in current_feature_names:
            if feature_name not in old_index:
                skipped += 1
                continue
            if patterns and not any(pattern in str(feature_name) for pattern in patterns):
                skipped += 1
                continue
            current_column = current_index[feature_name]
            old_column = old_index[feature_name]
            current_weight[:, current_column].mul_(1.0 - alpha).add_(source_weight[:, old_column], alpha=alpha)
            blended += 1

    return {
        "blended_features": blended,
        "skipped_features": skipped,
        "old_input_dim": len(old_feature_names),
        "current_input_dim": len(current_feature_names),
        "blend": alpha,
    }


def _point_chunks(batch: PointBatch, point_batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    point_count = batch.num_points
    if point_count <= 0:
        return []
    indices = torch.randperm(point_count) if shuffle else torch.arange(point_count)
    return [indices[start : start + point_batch_size] for start in range(0, point_count, point_batch_size)]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[PointBatch],
    optimizer: torch.optim.Optimizer,
    y_scaler: StandardScaler,
    device: torch.device,
    point_batch_size: int,
    grad_clip: float,
    loss_cfg: dict[str, Any] | None = None,
    logger: Any | None = None,
    epoch: int | None = None,
    progress_every_steps: int = 50,
) -> dict[str, float]:
    model.train()
    totals = _empty_point_metric_totals()
    started_at = time.monotonic()
    epoch_label = f"{epoch:04d}" if epoch is not None else "????"
    total_loader_steps = len(loader)
    total_data_wait_seconds = 0.0
    total_compute_seconds = 0.0
    next_batch_started_at = time.monotonic()

    for loader_step, host_batch in enumerate(loader, start=1):
        data_wait_seconds = time.monotonic() - next_batch_started_at
        total_data_wait_seconds += data_wait_seconds
        compute_started_at = time.monotonic()
        if host_batch.num_points <= 0:
            next_batch_started_at = time.monotonic()
            continue
        host_chunks = _point_chunks(host_batch, point_batch_size=point_batch_size, shuffle=True)
        for chunk in host_chunks:
            features = host_batch.features[chunk].to(device, non_blocking=True)
            target = host_batch.target_scaled[chunk].to(device, non_blocking=True)
            weights = host_batch.point_weights[chunk].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            prediction = regression_output(output)
            loss_values = F.smooth_l1_loss(prediction, target, reduction="none").squeeze(-1)
            loss = (loss_values * weights).sum() / weights.sum().clamp_min(1e-12)
            if loss_cfg:
                target_raw_chunk = host_batch.target_raw[chunk].to(device, non_blocking=True)
                sample_index_chunk = host_batch.sample_index[chunk].to(device, non_blocking=True)
                aux_loss = _compute_sample_tail_aux_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    sample_index=sample_index_chunk,
                    loss_cfg=loss_cfg,
                )
                disk_center_region_mask = (
                    host_batch.region_masks.get("disk_center_region")[chunk].to(device, non_blocking=True)
                    if host_batch.region_masks is not None and "disk_center_region" in host_batch.region_masks
                    else None
                )
                earpiece_region_mask = (
                    host_batch.region_masks.get("earpiece_region")[chunk].to(device, non_blocking=True)
                    if host_batch.region_masks is not None and "earpiece_region" in host_batch.region_masks
                    else None
                )
                aux_loss = aux_loss + _compute_region_tail_aux_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    sample_index=host_batch.sample_index[chunk].to(device, non_blocking=True),
                    region_mask=disk_center_region_mask,
                    loss_cfg=loss_cfg,
                    prefix="disk_center",
                )
                aux_loss = aux_loss + _compute_region_underprediction_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    sample_index=sample_index_chunk,
                    region_mask=disk_center_region_mask,
                    loss_cfg=loss_cfg,
                    y_scaler=y_scaler,
                    prefix="disk_center",
                )
                aux_loss = aux_loss + _compute_region_tail_aux_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    sample_index=sample_index_chunk,
                    region_mask=earpiece_region_mask,
                    loss_cfg=loss_cfg,
                    prefix="earpiece",
                )
                aux_loss = aux_loss + _compute_region_underprediction_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    sample_index=sample_index_chunk,
                    region_mask=earpiece_region_mask,
                    loss_cfg=loss_cfg,
                    y_scaler=y_scaler,
                    prefix="earpiece",
                )
                aux_loss = aux_loss + _compute_center_curve_consistency_loss(
                    prediction=prediction,
                    target=target,
                    sample_index=sample_index_chunk,
                    node_indices=host_batch.node_indices[chunk].to(device, non_blocking=True),
                    case_group=_case_group_tensor(host_batch.case_names, device),
                    region_mask=disk_center_region_mask,
                    loss_cfg=loss_cfg,
                    y_scaler=y_scaler,
                )
                low_target_loss = _compute_low_target_overprediction_loss(
                    prediction=prediction,
                    target=target,
                    target_raw=target_raw_chunk,
                    loss_cfg=loss_cfg,
                )
                censored_background_loss = _compute_censored_background_loss(
                    prediction=prediction,
                    target_raw=target_raw_chunk,
                    loss_cfg=loss_cfg,
                    y_scaler=y_scaler,
                )
                rank_loss = prediction.new_zeros(())
                rank_loss = (
                    rank_loss
                    + _compute_rank_band_balanced_loss(
                        prediction=prediction,
                        target=target,
                        target_raw=target_raw_chunk,
                        sample_index=sample_index_chunk,
                        loss_cfg=loss_cfg,
                    )
                    + _compute_participation_residual_loss(
                        prediction=prediction,
                        target=target,
                        target_raw=target_raw_chunk,
                        sample_index=sample_index_chunk,
                        loss_cfg=loss_cfg,
                        y_scaler=y_scaler,
                    )
                )
                if float(loss_cfg.get("background_false_peak_weight", 0.0)) > 0.0:
                    rank_loss = rank_loss + _compute_background_false_peak_loss(
                        prediction=prediction,
                        target=target,
                        target_raw=target_raw_chunk,
                        sample_index=sample_index_chunk,
                        loss_cfg=loss_cfg,
                        y_scaler=y_scaler,
                    )
                loss = loss + aux_loss + low_target_loss + censored_background_loss + rank_loss
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            point_count = int(chunk.numel())
            pred_log, pred_raw = _decode_prediction(prediction.detach(), y_scaler)
            _update_point_metric_totals(
                totals,
                loss_sum=loss_values.sum().item(),
                weighted_loss_sum=(loss_values * weights).sum().item(),
                weight_sum=weights.sum().item(),
                pred_log=pred_log,
                pred_raw=pred_raw,
                target_log=host_batch.target_log[chunk].to(device, non_blocking=True),
                target_raw=host_batch.target_raw[chunk].to(device, non_blocking=True),
            )

        total_compute_seconds += time.monotonic() - compute_started_at
        if logger is not None and (
            loader_step == 1
            or loader_step == total_loader_steps
            or (progress_every_steps > 0 and loader_step % progress_every_steps == 0)
        ):
            elapsed = time.monotonic() - started_at
            running_metrics = _finalize_point_metrics(totals)
            throughput = running_metrics["points"] / max(total_compute_seconds + total_data_wait_seconds, 1e-9)
            logger.info(
                (
                    "Epoch %s train progress | sample_batch=%s/%s | points=%s | elapsed=%s | "
                    "loss=%.6f | mae=%.6g | log_mae=%.6f | rel_mae=%.6f | "
                    "data_wait=%s | compute=%s | points/s=%.0f"
                ),
                epoch_label,
                loader_step,
                total_loader_steps,
                int(running_metrics["points"]),
                _format_duration(elapsed),
                running_metrics["loss"],
                running_metrics["earpiece_stress_mae"],
                running_metrics["earpiece_stress_log_mae"],
                running_metrics["earpiece_stress_relative_mae"],
                _format_duration(total_data_wait_seconds),
                _format_duration(total_compute_seconds),
                throughput,
            )
        next_batch_started_at = time.monotonic()
    return _finalize_point_metrics(totals)


def _decode_prediction(prediction_scaled: torch.Tensor, y_scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    y_scaler = y_scaler.to(prediction_scaled.device)
    pred_log = y_scaler.inverse_transform(prediction_scaled)
    pred_raw = torch.expm1(pred_log).clamp_min(0.0).squeeze(-1)
    return pred_log.squeeze(-1), pred_raw


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[PointBatch],
    y_scaler: StandardScaler,
    device: torch.device,
    point_batch_size: int,
    split_name: str,
    epoch: int | None = None,
    logger: Any | None = None,
    progress_every_steps: int = 50,
    collect_diagnostics: bool = False,
    topk_mode: str = "full",
    include_region_metrics: bool = True,
    max_sample_batches: int | None = None,
) -> EvaluationResult:
    topk_mode = str(topk_mode or "full").lower()
    if topk_mode in {"0", "false", "no", "none", "off"}:
        topk_mode = "none"
    elif topk_mode in {"selection", "basic", "fast"}:
        topk_mode = "selection"
    elif topk_mode != "full":
        raise ValueError(f"Unsupported evaluation topk_mode: {topk_mode}")
    if collect_diagnostics and topk_mode != "full":
        topk_mode = "full"

    include_basic_topk = topk_mode in {"selection", "full"}
    include_full_topk = topk_mode == "full"
    region_metric_specs = REGION_TOPK_METRIC_SPECS if include_region_metrics else []

    model.eval()
    totals = _empty_point_metric_totals()
    topk_totals = _empty_topk_metric_totals()
    region_totals = {prefix: _empty_point_metric_totals() for prefix, _ in region_metric_specs}
    region_topk_totals = {prefix: _empty_topk_metric_totals() for prefix, _ in region_metric_specs}
    top1_abs_sum = 0.0
    top1_log_abs_sum = 0.0
    top1_count = 0
    top5_abs_sum = 0.0
    top5_log_abs_sum = 0.0
    top5_count = 0
    peak_relative_error_sum = 0.0
    sample_count = 0
    diagnostics: list[dict[str, Any]] = []
    started_at = time.monotonic()
    total_loader_steps = len(loader)
    if max_sample_batches is not None and int(max_sample_batches) > 0:
        total_loader_steps = min(total_loader_steps, int(max_sample_batches))
    epoch_label = f"{epoch:04d}" if epoch is not None else "????"

    with torch.no_grad():
        loader_iter = islice(loader, total_loader_steps)
        for loader_step, host_batch in enumerate(loader_iter, start=1):
            if host_batch.num_points <= 0:
                continue
            predictions_scaled = []
            for chunk in _point_chunks(host_batch, point_batch_size=point_batch_size, shuffle=False):
                features = host_batch.features[chunk].to(device, non_blocking=True)
                predictions_scaled.append(regression_output(model(features)).detach())
            prediction_scaled = torch.cat(predictions_scaled, dim=0)
            target_scaled = host_batch.target_scaled.to(device, non_blocking=True)
            loss = F.smooth_l1_loss(prediction_scaled, target_scaled, reduction="sum").item()
            pred_log, pred_raw = _decode_prediction(prediction_scaled, y_scaler)
            target_log = host_batch.target_log.squeeze(-1).to(device, non_blocking=True)
            target_raw = host_batch.target_raw.to(device, non_blocking=True)
            sample_index = host_batch.sample_index.to(device, non_blocking=True)
            region_masks = (
                {
                    name: host_batch.region_masks[name].to(device, non_blocking=True).to(dtype=torch.bool)
                    for _, name in region_metric_specs
                    if host_batch.region_masks is not None and name in host_batch.region_masks
                }
                if region_metric_specs
                else {}
            )

            abs_error = (pred_raw - target_raw).abs()
            log_abs_error = (pred_log - target_log).abs()
            _update_point_metric_totals(
                totals,
                loss_sum=loss,
                pred_log=pred_log,
                pred_raw=pred_raw,
                target_log=target_log,
                target_raw=target_raw,
            )

            for sample_idx in range(len(host_batch.names)):
                mask = sample_index == sample_idx
                if not bool(mask.any()):
                    continue
                sample_target = target_raw[mask]
                sample_pred = pred_raw[mask]
                sample_target_log = target_log[mask]
                sample_pred_log = pred_log[mask]
                sample_abs = abs_error[mask]
                sample_delta = sample_pred - sample_target
                sample_log_delta = sample_pred_log - sample_target_log
                sample_relative_mask = sample_target.abs() > 1e-12
                sample_relative = torch.zeros_like(sample_abs)
                sample_relative[sample_relative_mask] = (
                    sample_abs[sample_relative_mask] / sample_target.abs()[sample_relative_mask]
                )
                sample_symmetric_relative = sample_abs / (
                    0.5 * (sample_pred.abs() + sample_target.abs())
                ).clamp_min(1e-12)
                target_peak = sample_target.max().item()
                pred_peak = sample_pred.max().item()
                peak_relative_error = abs(pred_peak - target_peak) / max(abs(target_peak), 1e-12)
                peak_relative_error_sum += peak_relative_error
                sample_count += 1

                sample_log_abs = log_abs_error[mask]
                top1_mask = torch.zeros_like(sample_target, dtype=torch.bool)
                top5_mask = torch.zeros_like(sample_target, dtype=torch.bool)
                sample_top1_mae = 0.0
                sample_top1_log_mae = 0.0
                sample_top5_mae = 0.0
                sample_top5_log_mae = 0.0
                sample_topk_masks: dict[str, torch.Tensor] = {}
                if include_basic_topk:
                    top1_threshold = torch.quantile(sample_target, 0.99)
                    top1_mask = sample_target >= top1_threshold
                    top1_abs_sum += sample_abs[top1_mask].sum().item()
                    top1_log_abs_sum += sample_log_abs[top1_mask].sum().item()
                    top1_count += int(top1_mask.sum().item())
                    sample_top1_mae = sample_abs[top1_mask].mean().item() if bool(top1_mask.any()) else 0.0
                    sample_top1_log_mae = sample_log_abs[top1_mask].mean().item() if bool(top1_mask.any()) else 0.0

                    top5_threshold = torch.quantile(sample_target, 0.95)
                    top5_mask = sample_target >= top5_threshold
                    top5_abs_sum += sample_abs[top5_mask].sum().item()
                    top5_log_abs_sum += sample_log_abs[top5_mask].sum().item()
                    top5_count += int(top5_mask.sum().item())
                    sample_top5_mae = sample_abs[top5_mask].mean().item() if bool(top5_mask.any()) else 0.0
                    sample_top5_log_mae = sample_log_abs[top5_mask].mean().item() if bool(top5_mask.any()) else 0.0

                    if include_full_topk:
                        sample_topk_masks = _rank_fraction_masks(sample_target)
                        _update_topk_metric_totals(
                            topk_totals,
                            pred_log=sample_pred_log,
                            pred_raw=sample_pred,
                            target_log=sample_target_log,
                            target_raw=sample_target,
                        )
                    else:
                        if bool(top1_mask.any()):
                            _update_point_metric_totals(
                                topk_totals["top1"],
                                loss_sum=0.0,
                                pred_log=sample_pred_log[top1_mask],
                                pred_raw=sample_pred[top1_mask],
                                target_log=sample_target_log[top1_mask],
                                target_raw=sample_target[top1_mask],
                            )
                        if bool(top5_mask.any()):
                            _update_point_metric_totals(
                                topk_totals["top5"],
                                loss_sum=0.0,
                                pred_log=sample_pred_log[top5_mask],
                                pred_raw=sample_pred[top5_mask],
                                target_log=sample_target_log[top5_mask],
                                target_raw=sample_target[top5_mask],
                            )

                for region_prefix, region_name in region_metric_specs:
                    region_mask_all = region_masks.get(region_name)
                    if region_mask_all is None:
                        continue
                    sample_region_mask = region_mask_all[mask]
                    if not bool(sample_region_mask.any()):
                        continue
                    _update_point_metric_totals(
                        region_totals[region_prefix],
                        loss_sum=0.0,
                        pred_log=sample_pred_log[sample_region_mask],
                        pred_raw=sample_pred[sample_region_mask],
                        target_log=sample_target_log[sample_region_mask],
                        target_raw=sample_target[sample_region_mask],
                    )
                    if include_full_topk:
                        _update_topk_metric_totals(
                            region_topk_totals[region_prefix],
                            pred_log=sample_pred_log[sample_region_mask],
                            pred_raw=sample_pred[sample_region_mask],
                            target_log=sample_target_log[sample_region_mask],
                            target_raw=sample_target[sample_region_mask],
                        )

                if collect_diagnostics:
                    sample_points = int(sample_target.numel())
                    sample_loss = F.smooth_l1_loss(
                        prediction_scaled[mask],
                        target_scaled[mask],
                        reduction="sum",
                    ).item() / max(sample_points, 1)
                    diagnostics.append(
                        {
                            "split": split_name,
                            "epoch": epoch if epoch is not None else "",
                            "sample_name": host_batch.names[sample_idx],
                            "case_name": host_batch.case_names[sample_idx],
                            "frequency_hz": float(host_batch.frequency_hz[sample_idx].item()),
                            "points": sample_points,
                            "loss": sample_loss,
                            "mae": sample_abs.mean().item(),
                            "rmse": sample_delta.pow(2).mean().sqrt().item(),
                            "log_mae": (sample_pred_log - sample_target_log).abs().mean().item(),
                            "log_rmse": sample_log_delta.pow(2).mean().sqrt().item(),
                            "bias": sample_delta.mean().item(),
                            "relative_mae": (
                                sample_relative[sample_relative_mask].mean().item()
                                if bool(sample_relative_mask.any())
                                else 0.0
                            ),
                            "symmetric_relative_mae": sample_symmetric_relative.mean().item(),
                            "within25_ratio": (
                                (sample_relative[sample_relative_mask] <= 0.25).to(torch.float32).mean().item()
                                if bool(sample_relative_mask.any())
                                else 0.0
                            ),
                            "target_mean": sample_target.mean().item(),
                            "pred_mean": sample_pred.mean().item(),
                            "target_peak": target_peak,
                            "pred_peak": pred_peak,
                            "peak_relative_error": peak_relative_error,
                            "top1_mae": sample_top1_mae,
                            "top1_log_mae": sample_top1_log_mae,
                            "top5_mae": sample_top5_mae,
                            "top5_log_mae": sample_top5_log_mae,
                            "top1_within25_ratio": _within25_ratio_for_mask(
                                sample_pred, sample_target, sample_topk_masks.get("top1", torch.zeros_like(sample_target, dtype=torch.bool))
                            ),
                            "top1_5_within25_ratio": _within25_ratio_for_mask(
                                sample_pred, sample_target, sample_topk_masks.get("top1_5", torch.zeros_like(sample_target, dtype=torch.bool))
                            ),
                            "top5_within25_ratio": _within25_ratio_for_mask(
                                sample_pred, sample_target, sample_topk_masks.get("top5", torch.zeros_like(sample_target, dtype=torch.bool))
                            ),
                            "top5_10_within25_ratio": _within25_ratio_for_mask(
                                sample_pred, sample_target, sample_topk_masks.get("top5_10", torch.zeros_like(sample_target, dtype=torch.bool))
                            ),
                            "top10_within25_ratio": _within25_ratio_for_mask(
                                sample_pred, sample_target, sample_topk_masks.get("top10", torch.zeros_like(sample_target, dtype=torch.bool))
                            ),
                            "under_pred_ratio": (sample_pred < sample_target).to(torch.float32).mean().item(),
                            "over_pred_ratio": (sample_pred > sample_target).to(torch.float32).mean().item(),
                        }
                    )

            if logger is not None and (
                loader_step == 1
                or loader_step == total_loader_steps
                or (progress_every_steps > 0 and loader_step % progress_every_steps == 0)
            ):
                running_metrics = _finalize_point_metrics(totals)
                logger.info(
                    (
                        "Epoch %s %s progress | sample_batch=%s/%s | points=%s | elapsed=%s | "
                        "loss=%.6f | mae=%.6g | log_mae=%.6f | rel_mae=%.6f"
                    ),
                    epoch_label,
                    split_name,
                    loader_step,
                    total_loader_steps,
                    int(running_metrics["points"]),
                    _format_duration(time.monotonic() - started_at),
                    running_metrics["loss"],
                    running_metrics["earpiece_stress_mae"],
                    running_metrics["earpiece_stress_log_mae"],
                    running_metrics["earpiece_stress_relative_mae"],
                )

    metrics = _finalize_point_metrics(totals)
    if include_basic_topk:
        metrics.update(
            {
                "earpiece_stress_top1_mae": top1_abs_sum / max(top1_count, 1),
                "earpiece_stress_top1_log_mae": top1_log_abs_sum / max(top1_count, 1),
                "earpiece_stress_top5_mae": top5_abs_sum / max(top5_count, 1),
                "earpiece_stress_top5_log_mae": top5_log_abs_sum / max(top5_count, 1),
                "earpiece_stress_peak_relative_error": peak_relative_error_sum / max(sample_count, 1),
                "top1_points": float(top1_count),
                "top5_points": float(top5_count),
                "samples": float(sample_count),
            }
        )
        topk_metrics = _finalize_topk_metric_totals(topk_totals)
        if not include_full_topk:
            topk_metrics = {
                key: value
                for key, value in topk_metrics.items()
                if (
                    (key.startswith("earpiece_stress_top1_") and not key.startswith("earpiece_stress_top1_5_"))
                    or (key.startswith("earpiece_stress_top5_") and not key.startswith("earpiece_stress_top5_10_"))
                )
            }
        metrics.update(topk_metrics)
    for region_prefix, _ in region_metric_specs:
        metrics.update(_finalize_point_metric_totals_with_prefix(region_totals[region_prefix], region_prefix))
        if include_full_topk:
            metrics.update(
                _finalize_topk_metric_totals_with_prefix(
                    region_topk_totals[region_prefix],
                    prefix=region_prefix,
                )
            )
    return EvaluationResult(metrics=metrics, diagnostics=diagnostics)


def _weighted_metric_sum(metrics: dict[str, float], selection_cfg: dict[str, Any]) -> float:
    components = selection_cfg.get("components", {})
    if not isinstance(components, dict) or not components:
        raise ValueError("Composite selection requires training.selection.components.")
    score = 0.0
    missing = []
    for metric_name, weight in components.items():
        if metric_name not in metrics:
            missing.append(metric_name)
            continue
        score += float(weight) * float(metrics[metric_name])
    if missing:
        raise KeyError(f"Composite selection metrics are unavailable: {missing}; available={sorted(metrics)}")
    return score


def compute_selection_score(metrics: dict[str, float], training_cfg: dict[str, Any]) -> tuple[str, float]:
    selection_cfg = dict(training_cfg.get("selection", {}))
    if selection_cfg:
        mode = str(selection_cfg.get("mode", "single")).lower()
        if mode in {"weighted_sum", "composite"}:
            return str(selection_cfg.get("name", "weighted_sum")), _weighted_metric_sum(metrics, selection_cfg)
        if mode not in {"single", "metric"}:
            raise ValueError(f"Unsupported training.selection.mode: {mode}")
        metric_name = str(selection_cfg.get("metric", training_cfg.get("selection_metric", "earpiece_stress_log_mae")))
        if metric_name not in metrics:
            raise KeyError(f"Selection metric '{metric_name}' is unavailable: {sorted(metrics)}")
        return metric_name, float(metrics[metric_name])

    metric_name = str(training_cfg.get("selection_metric", "earpiece_stress_log_mae"))
    if metric_name not in metrics:
        raise KeyError(f"Selection metric '{metric_name}' is unavailable: {sorted(metrics)}")
    return metric_name, float(metrics[metric_name])


def write_evaluation_diagnostics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=EVALUATION_DIAGNOSTIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_checkpoint(
    save_dir: Path,
    config: dict[str, Any],
    model: torch.nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "x_scaler": x_scaler.state_dict(),
            "y_scaler": y_scaler.state_dict(),
            "feature_schema": feature_schema,
            "metrics": metrics,
        },
        save_dir / "best.pt",
    )


def _load_scaler_cache(path: Path) -> tuple[StandardScaler, StandardScaler, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    return (
        StandardScaler.from_state_dict(payload["x_scaler"]),
        StandardScaler.from_state_dict(payload["y_scaler"]),
        dict(payload["feature_schema"]),
    )


def _save_scaler_cache(
    path: Path,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x_scaler": x_scaler.state_dict(),
            "y_scaler": y_scaler.state_dict(),
            "feature_schema": feature_schema,
            "config": config,
        },
        path,
    )


class NodeMLPTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.dataset_cfg = dict(config["dataset"])
        self.feature_cfg = dict(config.get("features", {}))
        self.training_cfg = dict(config["training"])
        self.scaler_cfg = dict(config.get("scaler", {}))
        self.target_cfg = dict(config.get("target", {}))
        self.loss_cfg = dict(config.get("loss", {}))

        self.save_dir = ensure_dir(self.training_cfg["save_dir"])
        self.logger = make_logger(self.save_dir, logger_name="case7_node_mlp")
        self.history_path = self.save_dir / "history.csv"

        self.case_index: dict[str, Path] = {}
        self.split_names: dict[str, list[str]] = {}
        self.train_sample_paths: list[Path] = []
        self.val_sample_paths: list[Path] = []
        self.test_sample_paths: list[Path] = []
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None
        self.feature_schema: dict[str, Any] = {}
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.resolved_config: dict[str, Any] = {}

        self._prepare()

    def _prepare(self) -> None:
        self.case_index = discover_case_index(self.dataset_cfg["root"])
        self.split_names = resolve_case_splits(self.dataset_cfg["root"], self.dataset_cfg)
        train_case_dirs = [self.case_index[name] for name in self.split_names["train"]]
        val_case_dirs = [self.case_index[name] for name in self.split_names["val"]]
        test_case_dirs = [self.case_index[name] for name in self.split_names.get("test", [])]

        self.train_sample_paths = expand_case_sample_paths(train_case_dirs, self.dataset_cfg)
        self.val_sample_paths = expand_case_sample_paths(val_case_dirs, self.dataset_cfg)
        self.test_sample_paths = expand_case_sample_paths(test_case_dirs, self.dataset_cfg)

        scaler_cache_path = self.scaler_cfg.get("cache_path")
        target_stats_cache_path = self.scaler_cfg.get("target_stats_cache_path", scaler_cache_path)
        cached_target_stats: dict[str, float] = {}
        if (
            bool(self.scaler_cfg.get("reuse_target_stats_from_cache", False))
            and target_stats_cache_path is not None
            and Path(target_stats_cache_path).exists()
        ):
            _cached_x, _cached_y, cached_schema = _load_scaler_cache(Path(target_stats_cache_path))
            cached_target_stats = {
                key: value
                for key, value in cached_schema.items()
                if str(key).startswith("target_") and isinstance(value, (int, float))
            }
            if cached_target_stats:
                self.logger.info("Reusing target stats from scaler cache: %s", target_stats_cache_path)

        target_stats = cached_target_stats or estimate_target_quantiles(
            train_sample_paths=self.train_sample_paths,
            dataset_cfg=self.dataset_cfg,
            sample_limit=self.scaler_cfg.get("target_stats_sample_limit", self.scaler_cfg.get("sample_limit")),
            sample_seed=int(self.scaler_cfg.get("sample_seed", self.training_cfg.get("seed", 42))),
            num_workers=int(self.scaler_cfg.get("num_workers", self.training_cfg.get("num_workers", 0))),
            max_values=self.scaler_cfg.get("target_stats_max_values", 2_000_000),
            logger=self.logger,
        )
        target_zero_threshold = _resolve_threshold_from_config(self.target_cfg, target_stats, "zero_below")
        target_stats["target_zero_threshold"] = float(target_zero_threshold)

        resolved_dataset = dict(self.dataset_cfg)
        resolved_dataset["split_mode"] = "explicit"
        resolved_dataset["train_cases"] = list(self.split_names["train"])
        resolved_dataset["val_cases"] = list(self.split_names["val"])
        resolved_dataset["test_cases"] = list(self.split_names.get("test", []))
        self.resolved_config = dict(self.config)
        self.resolved_config["dataset"] = resolved_dataset

        if scaler_cache_path is not None and Path(scaler_cache_path).exists():
            self.logger.info("Loading scaler cache: %s", scaler_cache_path)
            self.x_scaler, self.y_scaler, self.feature_schema = _load_scaler_cache(Path(scaler_cache_path))
            current_schema = _probe_current_feature_schema(self.train_sample_paths, self.dataset_cfg, self.feature_cfg)
            if not _scaler_cache_matches_schema(
                self.x_scaler,
                self.feature_schema,
                current_schema,
                target_zero_threshold=target_zero_threshold,
            ):
                self.logger.warning("Scaler cache feature schema mismatch; recomputing scalers.")
                self.x_scaler = None
                self.y_scaler = None
                self.feature_schema = {}
            else:
                self.feature_schema.update(target_stats)

        if self.x_scaler is None or self.y_scaler is None:
            self.x_scaler, self.y_scaler, self.feature_schema = fit_scalers(
                train_sample_paths=self.train_sample_paths,
                dataset_cfg=self.dataset_cfg,
                feature_cfg=self.feature_cfg,
                num_workers=int(self.scaler_cfg.get("num_workers", self.training_cfg.get("num_workers", 0))),
                sample_limit=self.scaler_cfg.get("sample_limit"),
                sample_seed=int(self.scaler_cfg.get("sample_seed", self.training_cfg.get("seed", 42))),
                prefetch_factor=int(self.scaler_cfg.get("prefetch_factor", 4)),
                target_zero_threshold=target_zero_threshold,
                target_stats=target_stats,
                logger=self.logger,
            )
            if scaler_cache_path is not None:
                if bool(self.scaler_cfg.get("overwrite_cache", False)):
                    self.logger.info("Saving scaler cache: %s", scaler_cache_path)
                    _save_scaler_cache(
                        Path(scaler_cache_path),
                        x_scaler=self.x_scaler,
                        y_scaler=self.y_scaler,
                        feature_schema=self.feature_schema,
                        config=self.resolved_config,
                    )
                else:
                    self.logger.info(
                        "Not saving recomputed scaler cache because scaler.overwrite_cache is false: %s",
                        scaler_cache_path,
                    )
        input_dim = int(self.feature_schema["input_dim"])
        self.model = build_model(self.config, input_dim=input_dim, feature_schema=self.feature_schema).to(self.device)
        init_checkpoint = self.training_cfg.get("init_checkpoint")
        if init_checkpoint:
            checkpoint_path = Path(str(init_checkpoint))
            if checkpoint_path.exists():
                self.logger.info("Initializing model from checkpoint: %s", checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_schema = dict(checkpoint.get("feature_schema", {}))
                if (
                    checkpoint_schema.get("feature_names") == self.feature_schema.get("feature_names")
                    and not bool(self.training_cfg.get("allow_partial_init_checkpoint", False))
                ):
                    self.model.load_state_dict(checkpoint["model_state"])
                elif bool(self.training_cfg.get("allow_partial_init_checkpoint", False)):
                    init_summary = _load_checkpoint_for_feature_schema_change(
                        self.model,
                        checkpoint=checkpoint,
                        current_feature_schema=self.feature_schema,
                    )
                    self.logger.info(
                        "Initialized checkpoint with feature schema changes: %s",
                        json.dumps(init_summary, ensure_ascii=False),
                    )
                else:
                    raise ValueError(
                        "training.init_checkpoint feature schema does not match current schema. "
                        "Set training.allow_partial_init_checkpoint=true to reuse shared feature weights."
                    )
            else:
                raise FileNotFoundError(f"training.init_checkpoint does not exist: {checkpoint_path}")
        aux_init_checkpoints = self.training_cfg.get("aux_init_checkpoints", [])
        if isinstance(aux_init_checkpoints, (str, Path)):
            aux_init_checkpoints = [{"path": str(aux_init_checkpoints)}]
        for aux_item in aux_init_checkpoints or []:
            if isinstance(aux_item, dict):
                aux_path = Path(str(aux_item.get("path", aux_item.get("checkpoint", ""))))
                blend = float(aux_item.get("blend", aux_item.get("weight", 0.25)))
                raw_patterns = aux_item.get("feature_patterns", aux_item.get("patterns", []))
                feature_patterns = (
                    [str(raw_patterns)]
                    if isinstance(raw_patterns, str)
                    else [str(pattern) for pattern in (raw_patterns or [])]
                )
            else:
                aux_path = Path(str(aux_item))
                blend = 0.25
                feature_patterns = []
            if not aux_path.exists():
                raise FileNotFoundError(f"training.aux_init_checkpoints path does not exist: {aux_path}")
            self.logger.info(
                "Blending first-layer feature weights from auxiliary checkpoint: %s | blend=%.3f",
                aux_path,
                blend,
            )
            aux_checkpoint = torch.load(aux_path, map_location="cpu")
            aux_summary = _blend_first_layer_from_checkpoint(
                self.model,
                checkpoint=aux_checkpoint,
                current_feature_schema=self.feature_schema,
                blend=blend,
                feature_patterns=feature_patterns,
            )
            self.logger.info(
                "Auxiliary first-layer blend summary: %s",
                json.dumps(aux_summary, ensure_ascii=False),
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("lr", 1e-3)),
            weight_decay=float(self.training_cfg.get("weight_decay", 1e-4)),
        )

        write_yaml(self.save_dir / "resolved_config.yaml", self.resolved_config)
        write_json(self.save_dir / "feature_schema.json", self.feature_schema)

    def _make_loader(self, sample_paths: list[Path], shuffle: bool, is_eval: bool = False) -> DataLoader[PointBatch]:
        assert self.x_scaler is not None and self.y_scaler is not None

        sample_batch_size = int(self.training_cfg.get("sample_batch_size", 1))
        num_workers = int(self.training_cfg.get("num_workers", 0))
        prefetch_factor = self.training_cfg.get("prefetch_factor")
        persistent_workers = bool(self.training_cfg.get("persistent_workers", False))

        if is_eval:
            sample_batch_size = int(self.training_cfg.get("eval_sample_batch_size", sample_batch_size))
            num_workers = int(self.training_cfg.get("eval_num_workers", num_workers))
            if "eval_prefetch_factor" in self.training_cfg:
                prefetch_factor = self.training_cfg.get("eval_prefetch_factor")
            if "eval_persistent_workers" in self.training_cfg:
                persistent_workers = bool(self.training_cfg.get("eval_persistent_workers"))

        return make_loader(
            sample_paths=sample_paths,
            dataset_cfg=self.dataset_cfg,
            feature_cfg=self.feature_cfg,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            feature_schema=self.feature_schema,
            target_cfg=self.target_cfg,
            loss_cfg=self.loss_cfg,
            sample_batch_size=sample_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=self.training_cfg.get("pin_memory"),
            cache_prepared_samples=bool(self.training_cfg.get("cache_prepared_samples", False)),
            max_cached_samples_per_worker=self.training_cfg.get("max_cached_samples_per_worker"),
        )

    def _append_history_row(self, row: dict[str, Any]) -> None:
        existing_rows: list[dict[str, Any]] = []
        existing_fields: list[str] = []
        if self.history_path.exists():
            with self.history_path.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)

        fieldnames = list(HISTORY_BASE_FIELDS)
        for name in [*existing_fields, *row.keys()]:
            if name not in fieldnames:
                fieldnames.append(name)

        existing_rows.append(row)
        with self.history_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for item in existing_rows:
                writer.writerow(item)

    def fit(self) -> dict[str, Any]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.x_scaler is not None and self.y_scaler is not None

        self.logger.info("Device: %s", self.device)
        self.logger.info(
            "Train/Val/Test samples: %s/%s/%s",
            len(self.train_sample_paths),
            len(self.val_sample_paths),
            len(self.test_sample_paths),
        )
        self.logger.info("Feature schema: %s", json.dumps(self.feature_schema, ensure_ascii=False))
        self.logger.info("Target config: %s", json.dumps(self.target_cfg, ensure_ascii=False))
        self.logger.info("Loss config: %s", json.dumps(self.loss_cfg, ensure_ascii=False))
        self.logger.info("Training DataLoader workers: %s", int(self.training_cfg.get("num_workers", 0)))

        selection_label = str(
            self.training_cfg.get(
                "selection_metric",
                dict(self.training_cfg.get("selection", {})).get("name", "earpiece_stress_log_mae"),
            )
        )
        best_score = float("inf")
        best_payload: dict[str, Any] | None = None
        wait = 0
        patience = int(self.training_cfg.get("early_stopping_patience", 8))
        point_batch_size = int(self.training_cfg.get("batch_size", 32768))
        grad_clip = float(self.training_cfg.get("grad_clip", 1.0))
        eval_every = int(self.training_cfg.get("eval_every", 1))
        progress_every_steps = int(self.training_cfg.get("progress_every_steps", 50))
        eval_progress_every_steps = int(self.training_cfg.get("eval_progress_every_steps", progress_every_steps))
        write_diagnostics = bool(self.training_cfg.get("write_diagnostics", True))
        eval_topk_mode = str(self.training_cfg.get("eval_topk_mode", "full"))
        eval_include_region_metrics = bool(self.training_cfg.get("eval_include_region_metrics", True))
        eval_max_sample_batches = self.training_cfg.get("eval_max_sample_batches")
        eval_max_sample_batches = (
            int(eval_max_sample_batches)
            if eval_max_sample_batches is not None and int(eval_max_sample_batches) > 0
            else None
        )
        test_on_best = bool(self.training_cfg.get("test_on_best", True))
        final_test_after_training = bool(self.training_cfg.get("final_test_after_training", not test_on_best))
        final_test_topk_mode = str(self.training_cfg.get("final_test_topk_mode", "full"))
        final_test_include_region_metrics = bool(self.training_cfg.get("final_test_include_region_metrics", True))
        final_test_max_sample_batches = self.training_cfg.get("final_test_max_sample_batches")
        final_test_max_sample_batches = (
            int(final_test_max_sample_batches)
            if final_test_max_sample_batches is not None and int(final_test_max_sample_batches) > 0
            else None
        )

        for epoch in range(1, int(self.training_cfg.get("epochs", 30)) + 1):
            train_loader = self._make_loader(
                self.train_sample_paths,
                shuffle=bool(self.training_cfg.get("shuffle_samples", False)),
            )
            train_metrics = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                y_scaler=self.y_scaler,
                device=self.device,
                point_batch_size=point_batch_size,
                grad_clip=grad_clip,
                loss_cfg=self.loss_cfg,
                logger=self.logger,
                epoch=epoch,
                progress_every_steps=progress_every_steps,
            )
            train_loss = float(train_metrics["loss"])
            history_row: dict[str, Any] = {"epoch": epoch, "train_loss": round(float(train_loss), 8)}
            for key, value in train_metrics.items():
                if key not in {"loss", "weighted_loss"}:
                    history_row[f"train_{key}"] = round(float(value), 8)
            history_row["train_weighted_loss"] = round(float(train_metrics.get("weighted_loss", train_loss)), 8)

            if epoch % eval_every != 0:
                self._append_history_row(history_row)
                self.logger.info(
                    "Epoch %04d | train=%s | eval=skipped",
                    epoch,
                    json.dumps(train_metrics, ensure_ascii=False),
                )
                continue

            val_loader = self._make_loader(self.val_sample_paths, shuffle=False, is_eval=True)
            val_result = evaluate(
                model=self.model,
                loader=val_loader,
                y_scaler=self.y_scaler,
                device=self.device,
                point_batch_size=point_batch_size,
                split_name="val",
                epoch=epoch,
                logger=self.logger,
                progress_every_steps=eval_progress_every_steps,
                collect_diagnostics=write_diagnostics,
                topk_mode=eval_topk_mode,
                include_region_metrics=eval_include_region_metrics,
                max_sample_batches=eval_max_sample_batches,
            )
            val_metrics = val_result.metrics
            history_row["val_loss"] = round(float(val_metrics["loss"]), 8)
            for key, value in val_metrics.items():
                if key not in {"loss", "weighted_loss"}:
                    history_row[f"val_{key}"] = round(float(value), 8)
            history_row["val_weighted_loss"] = round(float(val_metrics.get("weighted_loss", val_metrics["loss"])), 8)

            selection_label, score = compute_selection_score(val_metrics, self.training_cfg)
            history_row["selection_metric"] = selection_label
            history_row["selection_score"] = round(float(score), 8)
            self._append_history_row(history_row)
            self.logger.info(
                "Epoch %04d | train=%s | val=%s",
                epoch,
                json.dumps(train_metrics, ensure_ascii=False),
                json.dumps(val_metrics, ensure_ascii=False),
            )

            if score < best_score:
                best_score = score
                wait = 0
                test_metrics: dict[str, float] = {}
                test_result = EvaluationResult(metrics={}, diagnostics=[])
                if test_on_best and self.test_sample_paths:
                    test_loader = self._make_loader(self.test_sample_paths, shuffle=False, is_eval=True)
                    test_result = evaluate(
                        model=self.model,
                        loader=test_loader,
                        y_scaler=self.y_scaler,
                        device=self.device,
                        point_batch_size=point_batch_size,
                        split_name="test",
                        epoch=epoch,
                        logger=self.logger,
                        progress_every_steps=eval_progress_every_steps,
                        collect_diagnostics=write_diagnostics,
                        topk_mode=final_test_topk_mode,
                        include_region_metrics=final_test_include_region_metrics,
                        max_sample_batches=final_test_max_sample_batches,
                    )
                    test_metrics = test_result.metrics
                if write_diagnostics:
                    write_evaluation_diagnostics(self.save_dir / "best_val_diagnostics.csv", val_result.diagnostics)
                    if test_on_best and self.test_sample_paths:
                        write_evaluation_diagnostics(self.save_dir / "best_test_diagnostics.csv", test_result.diagnostics)
                best_payload = {
                    "epoch": epoch,
                    "selection_metric": selection_label,
                    "selection_score": score,
                    "train_loss": train_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
                _save_checkpoint(
                    save_dir=self.save_dir,
                    config=self.resolved_config,
                    model=self.model,
                    x_scaler=self.x_scaler,
                    y_scaler=self.y_scaler,
                    feature_schema=self.feature_schema,
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
        if final_test_after_training and (not test_on_best) and self.test_sample_paths:
            checkpoint_path = self.save_dir / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state"])
            self.logger.info("Running final test for best checkpoint: %s", checkpoint_path)
            test_loader = self._make_loader(self.test_sample_paths, shuffle=False, is_eval=True)
            test_result = evaluate(
                model=self.model,
                loader=test_loader,
                y_scaler=self.y_scaler,
                device=self.device,
                point_batch_size=point_batch_size,
                split_name="test",
                epoch=int(best_payload["epoch"]),
                logger=self.logger,
                progress_every_steps=eval_progress_every_steps,
                collect_diagnostics=write_diagnostics,
                topk_mode=final_test_topk_mode,
                include_region_metrics=final_test_include_region_metrics,
                max_sample_batches=final_test_max_sample_batches,
            )
            best_payload["test_metrics"] = test_result.metrics
            checkpoint["metrics"] = best_payload
            torch.save(checkpoint, checkpoint_path)
            write_json(self.save_dir / "metrics.json", best_payload)
            if write_diagnostics:
                write_evaluation_diagnostics(self.save_dir / "best_test_diagnostics.csv", test_result.diagnostics)
        self.logger.info("Best run summary:\n%s", json.dumps(best_payload, indent=2, ensure_ascii=False))
        return best_payload

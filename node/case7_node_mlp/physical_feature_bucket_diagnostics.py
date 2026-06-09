from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import argparse
import csv
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import PointMLP
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, make_loader


QUANTILE_LABELS = ("p0_50", "p50_80", "p80_95", "p95_99", "p99_100")
QUANTILE_LEVELS = (0.50, 0.80, 0.95, 0.99)
REGION_LABELS = (
    "background_non_stress_region",
    "center_couple_region",
    "plate_hole_region",
    "ear_hole_region",
    "ear_connection_region",
    "bc_region",
)
REGION_COLUMNS = {
    "center_couple_region": ("center_couple_mask",),
    "plate_hole_region": ("plate_hole_wall_mask",),
    "ear_hole_region": ("ear_hole_wall_mask",),
    "ear_connection_region": (
        "ear_connection_fillet_mask",
        "ear_connection_earside_mask",
        "ear_connection_mask",
    ),
    "bc_region": ("bc_mask",),
}


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    feature_name: str
    bucket_kind: str
    fixed_edges: tuple[float, ...] = ()
    fixed_labels: tuple[str, ...] = ()


FEATURE_SPECS = (
    FeatureSpec("frf_gain", "active1_log_modal_gain_frf", "quantile"),
    FeatureSpec(
        "modal_weight",
        "active1_modal_weight_frf",
        "fixed",
        (-float("inf"), 0.40, 0.70, 0.90, float("inf")),
        ("lt_0p40", "0p40_0p70", "0p70_0p90", "ge_0p90"),
    ),
    FeatureSpec("weighted_umag", "weighted_umag_frf", "quantile"),
    FeatureSpec("weighted_gradient", "weighted_grad_umag_max_frf", "quantile"),
    FeatureSpec("modal_baseline", "modal_baseline_log_grad_umag_max", "quantile"),
    FeatureSpec("log_psd", "log_psd_value_at_frequency", "quantile"),
    FeatureSpec(
        "center_radius",
        "center_region_r_over_mask_radius",
        "fixed",
        (-float("inf"), 1.0, 2.0, 4.0, float("inf")),
        ("le_1", "1_2", "2_4", "gt_4"),
    ),
    FeatureSpec(
        "plate_hole_edge",
        "dist_to_plate_hole_edge_over_radius",
        "fixed",
        (-float("inf"), 0.0, 0.50, 1.0, 2.0, float("inf")),
        ("inside_hole", "0_0p5", "0p5_1", "1_2", "gt_2"),
    ),
)

CROSS_SPECS = (
    ("frf_gain_x_weighted_umag", "frf_gain", "weighted_umag"),
    ("frf_gain_x_weighted_gradient", "frf_gain", "weighted_gradient"),
    ("modal_weight_x_weighted_umag", "modal_weight", "weighted_umag"),
    ("log_psd_x_frf_gain", "log_psd", "frf_gain"),
    ("region_x_modal_baseline", "region", "modal_baseline"),
    ("center_radius_x_modal_baseline", "center_radius", "modal_baseline"),
    ("plate_hole_edge_x_modal_baseline", "plate_hole_edge", "modal_baseline"),
)


@dataclass
class BucketStats:
    points: int = 0
    relative_points: int = 0
    within25_count: int = 0
    abs_sum: float = 0.0
    log_abs_sum: float = 0.0
    relative_sum: float = 0.0
    under_count: int = 0
    over_count: int = 0
    target_sum: float = 0.0
    pred_sum: float = 0.0

    def update(
        self,
        target_raw: np.ndarray,
        pred_raw: np.ndarray,
        target_log: np.ndarray,
        pred_log: np.ndarray,
    ) -> None:
        if target_raw.size == 0:
            return
        delta = pred_raw - target_raw
        abs_error = np.abs(delta).astype(np.float64, copy=False)
        log_abs = np.abs(pred_log - target_log).astype(np.float64, copy=False)
        relative_mask = np.abs(target_raw) > 1e-12
        relative = abs_error[relative_mask] / np.abs(target_raw[relative_mask]).astype(np.float64, copy=False)

        self.points += int(target_raw.size)
        self.relative_points += int(relative_mask.sum())
        self.within25_count += int((relative <= 0.25).sum())
        self.abs_sum += float(abs_error.sum())
        self.log_abs_sum += float(log_abs.sum())
        self.relative_sum += float(relative.sum())
        self.under_count += int((pred_raw < target_raw).sum())
        self.over_count += int((pred_raw > target_raw).sum())
        self.target_sum += float(target_raw.astype(np.float64, copy=False).sum())
        self.pred_sum += float(pred_raw.astype(np.float64, copy=False).sum())

    def metrics(self) -> dict[str, Any]:
        points = max(self.points, 1)
        relative_points = max(self.relative_points, 1)
        within25 = self.within25_count / relative_points
        return {
            "points": self.points,
            "relative_points": self.relative_points,
            "within25_ratio": within25,
            "miss25_rate": 1.0 - within25,
            "mae": self.abs_sum / points,
            "log_mae": self.log_abs_sum / points,
            "relative_mae": self.relative_sum / relative_points,
            "under_pred_ratio": self.under_count / points,
            "over_pred_ratio": self.over_count / points,
            "target_mean": self.target_sum / points,
            "pred_mean": self.pred_sum / points,
            "pred_target_ratio": self.pred_sum / max(self.target_sum, 1e-12),
        }


class FeatureAccessor:
    def __init__(self, feature_schema: dict[str, Any], x_scaler: StandardScaler) -> None:
        self.feature_names = list(feature_schema["feature_names"])
        self.geometry_names = list(feature_schema.get("geometry_normalized_feature_names", []))
        self.scaled_names = list(feature_schema.get("scaled_continuous_feature_names", []))
        self.mask_names = list(feature_schema.get("mask_feature_names", []))
        expected_names = self.geometry_names + self.scaled_names + self.mask_names
        if expected_names != self.feature_names:
            raise ValueError("Checkpoint feature schema ordering is inconsistent.")
        if int(x_scaler.mean.numel()) != len(self.scaled_names):
            raise ValueError("Scaler dimension does not match scaled feature names.")
        self.global_indices = {name: idx for idx, name in enumerate(self.feature_names)}
        self.scaled_indices = {name: idx for idx, name in enumerate(self.scaled_names)}
        self.scaler_mean = x_scaler.mean.detach().cpu().numpy().astype(np.float32, copy=False)
        self.scaler_std = x_scaler.std.detach().cpu().numpy().astype(np.float32, copy=False)

    def validate(self, feature_names: list[str]) -> None:
        missing = [name for name in feature_names if name not in self.global_indices]
        if missing:
            raise ValueError(f"Checkpoint is missing requested features: {missing}")

    def values(self, features: torch.Tensor, feature_name: str) -> np.ndarray:
        values = features[:, self.global_indices[feature_name]].detach().cpu().numpy().astype(np.float32, copy=False)
        scaled_index = self.scaled_indices.get(feature_name)
        if scaled_index is not None:
            values = values * self.scaler_std[scaled_index] + self.scaler_mean[scaled_index]
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket diagnostics for physical input features in a node MLP.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fit-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument(
        "--dataset-max-cases",
        type=int,
        default=None,
        help="Apply a case limit before the configured train/val/test ratio split.",
    )
    parser.add_argument("--fit-cases", type=int, default=10)
    parser.add_argument("--eval-cases", type=int, default=10)
    parser.add_argument("--case-seed", type=int, default=42)
    parser.add_argument("--max-frames-per-case", type=int, default=100)
    parser.add_argument("--max-quantile-values", type=int, default=2_000_000)
    return parser.parse_args()


def _select_names(names: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit <= 0 or limit >= len(names):
        return list(names)
    selected = random.Random(seed).sample(list(names), k=int(limit))
    return sorted(selected)


def _make_loader(
    *,
    split: str,
    case_limit: int | None,
    case_seed: int,
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    feature_schema: dict[str, Any],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    sample_batch_size: int | None,
    num_workers: int | None,
) -> tuple[Any, dict[str, Path], list[str], int]:
    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    selected_names = _select_names(list(split_names[split]), case_limit, seed=case_seed)
    case_dirs = [case_index[name] for name in selected_names]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    loader = make_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=int(sample_batch_size or training_cfg.get("sample_batch_size", 1)),
        num_workers=int(num_workers if num_workers is not None else training_cfg.get("num_workers", 0)),
        shuffle=False,
    )
    return loader, case_index, selected_names, len(sample_paths)


def _case_region_arrays(case_dir: Path) -> dict[str, np.ndarray]:
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    arrays: dict[str, np.ndarray] = {}
    stress_any = np.zeros(len(nodes_df), dtype=bool)
    for label, columns in REGION_COLUMNS.items():
        mask = np.zeros(len(nodes_df), dtype=bool)
        for column in columns:
            if column in nodes_df.columns:
                mask |= nodes_df[column].to_numpy(dtype=np.float32) > 0.5
        arrays[label] = mask
        if label != "bc_region":
            stress_any |= mask
    arrays["background_non_stress_region"] = ~stress_any & ~arrays.get(
        "bc_region",
        np.zeros(len(nodes_df), dtype=bool),
    )
    return arrays


def _exclusive_region_ids(case_regions: dict[str, np.ndarray], node_rows: np.ndarray) -> np.ndarray:
    ids = np.zeros(node_rows.shape[0], dtype=np.int16)
    for label in (
        "center_couple_region",
        "plate_hole_region",
        "ear_hole_region",
        "ear_connection_region",
        "bc_region",
    ):
        mask = case_regions.get(label, np.zeros(0, dtype=bool))[node_rows]
        ids[mask] = REGION_LABELS.index(label)
    return ids


def _sample_values(values: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if cap <= 0 or finite.size <= cap:
        return finite.astype(np.float32, copy=False)
    indices = rng.choice(finite.size, size=cap, replace=False)
    return finite[indices].astype(np.float32, copy=False)


def _fit_quantile_edges(
    loader: Any,
    accessor: FeatureAccessor,
    max_values: int,
    seed: int,
    logger: Any,
) -> dict[str, list[float]]:
    quantile_specs = [spec for spec in FEATURE_SPECS if spec.bucket_kind == "quantile"]
    values_by_key: dict[str, list[np.ndarray]] = {spec.key: [] for spec in quantile_specs}
    per_batch_cap = max(1, int(max_values) // max(len(loader), 1)) if max_values > 0 else 0
    rng = np.random.default_rng(seed)
    sampled_counts = {spec.key: 0 for spec in quantile_specs}

    for loader_step, batch in enumerate(loader, start=1):
        for spec in quantile_specs:
            remaining = max(0, int(max_values) - sampled_counts[spec.key]) if max_values > 0 else 0
            if max_values > 0 and remaining <= 0:
                continue
            cap = min(per_batch_cap, remaining) if max_values > 0 else 0
            sampled = _sample_values(accessor.values(batch.features, spec.feature_name), cap, rng)
            if sampled.size:
                values_by_key[spec.key].append(sampled)
                sampled_counts[spec.key] += int(sampled.size)
        if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
            logger.info(
                "quantile fit | sample_batch=%s/%s | sampled_per_feature=%s",
                loader_step,
                len(loader),
                min(sampled_counts.values()) if sampled_counts else 0,
            )

    edges: dict[str, list[float]] = {}
    for spec in quantile_specs:
        if not values_by_key[spec.key]:
            raise ValueError(f"No finite values collected for feature: {spec.feature_name}")
        values = np.concatenate(values_by_key[spec.key]).astype(np.float64, copy=False)
        thresholds = np.quantile(values, np.asarray(QUANTILE_LEVELS, dtype=np.float64))
        edges[spec.key] = [-float("inf"), *[float(value) for value in thresholds], float("inf")]
    return edges


def _bucket_definition(
    spec: FeatureSpec,
    quantile_edges: dict[str, list[float]],
) -> tuple[list[float], list[str]]:
    if spec.bucket_kind == "quantile":
        return list(quantile_edges[spec.key]), list(QUANTILE_LABELS)
    return list(spec.fixed_edges), list(spec.fixed_labels)


def _bucket_ids(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.clip(
        np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right") - 1,
        0,
        len(edges) - 2,
    ).astype(np.int16, copy=False)


def _update_stats_by_ids(
    stats: dict[int, BucketStats],
    ids: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    for bucket_id in np.unique(ids):
        mask = ids == bucket_id
        stats.setdefault(int(bucket_id), BucketStats()).update(
            target_raw[mask],
            pred_raw[mask],
            target_log[mask],
            pred_log[mask],
        )


def _update_cross_stats(
    stats: dict[tuple[int, int], BucketStats],
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    target_log: np.ndarray,
    pred_log: np.ndarray,
) -> None:
    combined = np.stack([row_ids, col_ids], axis=1)
    for row_id, col_id in np.unique(combined, axis=0):
        mask = (row_ids == row_id) & (col_ids == col_id)
        stats.setdefault((int(row_id), int(col_id)), BucketStats()).update(
            target_raw[mask],
            pred_raw[mask],
            target_log[mask],
            pred_log[mask],
        )


def _format_edge(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8g}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device) -> PointMLP:
    feature_schema = dict(checkpoint["feature_schema"])
    model_cfg = dict(config.get("model", {}))
    model = PointMLP(
        input_dim=int(feature_schema["input_dim"]),
        hidden_dims=[int(dim) for dim in model_cfg.get("hidden_dims", [256, 256, 128])],
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "silu")),
        use_layer_norm=bool(model_cfg.get("layer_norm", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    training_cfg = dict(config.get("training", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    if args.dataset_max_cases is not None:
        dataset_cfg["max_cases"] = int(args.dataset_max_cases)
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(
        args.output_dir
        or checkpoint_path.parent / f"physical_feature_buckets_{args.fit_split}_to_{args.eval_split}"
    )
    logger = make_logger(
        output_dir,
        logger_name="case7_node_mlp.physical_feature_bucket_diagnostics",
        log_file="physical_feature_bucket_diagnostics.log",
    )
    device = resolve_device(args.device)
    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    accessor = FeatureAccessor(feature_schema, x_scaler)
    accessor.validate([spec.feature_name for spec in FEATURE_SPECS])

    fit_loader, _, fit_names, fit_samples = _make_loader(
        split=args.fit_split,
        case_limit=args.fit_cases,
        case_seed=int(args.case_seed),
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        training_cfg=training_cfg,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        feature_schema=feature_schema,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sample_batch_size=args.sample_batch_size,
        num_workers=args.num_workers,
    )
    eval_loader, case_index, eval_names, eval_samples = _make_loader(
        split=args.eval_split,
        case_limit=args.eval_cases,
        case_seed=int(args.case_seed) + 1,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        training_cfg=training_cfg,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        feature_schema=feature_schema,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        sample_batch_size=args.sample_batch_size,
        num_workers=args.num_workers,
    )
    logger.info(
        "Physical feature buckets | fit=%s cases=%s samples=%s | eval=%s cases=%s samples=%s | device=%s",
        args.fit_split,
        len(fit_names),
        fit_samples,
        args.eval_split,
        len(eval_names),
        eval_samples,
        device,
    )

    quantile_edges = _fit_quantile_edges(
        fit_loader,
        accessor,
        max_values=int(args.max_quantile_values),
        seed=int(args.case_seed),
        logger=logger,
    )
    definitions = {
        spec.key: _bucket_definition(spec, quantile_edges)
        for spec in FEATURE_SPECS
    }
    definitions["region"] = (
        list(range(len(REGION_LABELS) + 1)),
        list(REGION_LABELS),
    )

    model = _load_model(checkpoint, config, device)
    feature_stats: dict[str, dict[int, BucketStats]] = {
        spec.key: {} for spec in FEATURE_SPECS
    }
    feature_stats["region"] = {}
    cross_stats: dict[str, dict[tuple[int, int], BucketStats]] = {
        name: {} for name, _, _ in CROSS_SPECS
    }
    region_cache: dict[str, dict[str, np.ndarray]] = {}
    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    evaluated_points = 0

    with torch.no_grad():
        for loader_step, host_batch in enumerate(eval_loader, start=1):
            prediction_parts: list[torch.Tensor] = []
            for start in range(0, host_batch.num_points, point_batch_size):
                end = min(start + point_batch_size, host_batch.num_points)
                prediction_parts.append(model(host_batch.features[start:end].to(device, non_blocking=True)).cpu())
            prediction_scaled = torch.cat(prediction_parts, dim=0).to(device)
            pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled, y_scaler)
            pred_log = pred_log_t.cpu().numpy().astype(np.float32, copy=False)
            pred_raw = pred_raw_t.cpu().numpy().astype(np.float32, copy=False)
            target_log = host_batch.target_log.squeeze(-1).numpy().astype(np.float32, copy=False)
            target_raw = host_batch.target_raw.numpy().astype(np.float32, copy=False)
            node_rows = host_batch.node_indices.numpy().astype(np.int64, copy=False)
            sample_indices = host_batch.sample_index.numpy().astype(np.int64, copy=False)

            ids_by_key: dict[str, np.ndarray] = {}
            for spec in FEATURE_SPECS:
                edges, _ = definitions[spec.key]
                ids_by_key[spec.key] = _bucket_ids(
                    accessor.values(host_batch.features, spec.feature_name),
                    edges,
                )
                _update_stats_by_ids(
                    feature_stats[spec.key],
                    ids_by_key[spec.key],
                    target_raw,
                    pred_raw,
                    target_log,
                    pred_log,
                )

            region_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            for sample_idx, case_name in enumerate(host_batch.case_names):
                sample_mask = sample_indices == sample_idx
                if case_name not in region_cache:
                    region_cache[case_name] = _case_region_arrays(case_index[case_name])
                region_ids[sample_mask] = _exclusive_region_ids(
                    region_cache[case_name],
                    node_rows[sample_mask],
                )
            ids_by_key["region"] = region_ids
            _update_stats_by_ids(
                feature_stats["region"],
                region_ids,
                target_raw,
                pred_raw,
                target_log,
                pred_log,
            )

            for cross_name, row_key, col_key in CROSS_SPECS:
                _update_cross_stats(
                    cross_stats[cross_name],
                    ids_by_key[row_key],
                    ids_by_key[col_key],
                    target_raw,
                    pred_raw,
                    target_log,
                    pred_log,
                )

            evaluated_points += int(target_raw.size)
            if loader_step == 1 or loader_step == len(eval_loader) or loader_step % 20 == 0:
                logger.info(
                    "evaluation | sample_batch=%s/%s | points=%s",
                    loader_step,
                    len(eval_loader),
                    evaluated_points,
                )

    feature_rows: list[dict[str, Any]] = []
    for key, stats_by_id in feature_stats.items():
        edges, labels = definitions[key]
        for bucket_id, label in enumerate(labels):
            stats = stats_by_id.get(bucket_id, BucketStats())
            feature_rows.append(
                {
                    "feature_key": key,
                    "feature_name": next(
                        (spec.feature_name for spec in FEATURE_SPECS if spec.key == key),
                        "region",
                    ),
                    "bucket_id": bucket_id,
                    "bucket": label,
                    "lower": _format_edge(float(edges[bucket_id])),
                    "upper": _format_edge(float(edges[bucket_id + 1])),
                    **stats.metrics(),
                }
            )
    feature_csv = output_dir / "physical_feature_buckets.csv"
    _write_csv(feature_csv, feature_rows)

    cross_rows: list[dict[str, Any]] = []
    for cross_name, row_key, col_key in CROSS_SPECS:
        _, row_labels = definitions[row_key]
        _, col_labels = definitions[col_key]
        for (row_id, col_id), stats in sorted(cross_stats[cross_name].items()):
            cross_rows.append(
                {
                    "cross": cross_name,
                    "row_feature": row_key,
                    "row_bucket_id": row_id,
                    "row_bucket": row_labels[row_id],
                    "col_feature": col_key,
                    "col_bucket_id": col_id,
                    "col_bucket": col_labels[col_id],
                    **stats.metrics(),
                }
            )
    cross_csv = output_dir / "physical_feature_cross_buckets.csv"
    _write_csv(cross_csv, cross_rows)

    edge_payload = {
        spec.key: {
            "feature_name": spec.feature_name,
            "bucket_kind": spec.bucket_kind,
            "edges": [_format_edge(float(value)) for value in definitions[spec.key][0]],
            "labels": definitions[spec.key][1],
        }
        for spec in FEATURE_SPECS
    }
    write_json(output_dir / "physical_feature_bucket_edges.json", edge_payload)
    summary = {
        "checkpoint": str(checkpoint_path),
        "fit_split": args.fit_split,
        "fit_cases": fit_names,
        "fit_samples": fit_samples,
        "eval_split": args.eval_split,
        "eval_cases": eval_names,
        "eval_samples": eval_samples,
        "evaluated_points": evaluated_points,
        "dataset_max_cases": args.dataset_max_cases,
        "max_frames_per_case": args.max_frames_per_case,
        "features": [spec.key for spec in FEATURE_SPECS],
        "crosses": [name for name, _, _ in CROSS_SPECS],
        "outputs": {
            "feature_buckets": str(feature_csv),
            "cross_buckets": str(cross_csv),
            "bucket_edges": str(output_dir / "physical_feature_bucket_edges.json"),
        },
    }
    write_json(output_dir / "physical_feature_bucket_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

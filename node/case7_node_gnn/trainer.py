from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from case7_node_gnn.data import GraphBatch, make_graph_loader
from case7_node_gnn.models import NodeTransformerConv, _require_transformer_conv
from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.runtime import ensure_dir, make_logger, write_json, write_yaml
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import (
    EvaluationResult,
    HISTORY_BASE_FIELDS,
    _decode_prediction,
    _empty_point_metric_totals,
    _empty_topk_metric_totals,
    _finalize_point_metrics,
    _finalize_topk_metric_totals,
    _format_duration,
    _load_scaler_cache,
    _probe_current_feature_schema,
    _rank_fraction_masks,
    _resolve_threshold_from_config,
    _save_checkpoint,
    _save_scaler_cache,
    _scaler_cache_matches_schema,
    _update_topk_metric_totals,
    _update_point_metric_totals,
    _within25_ratio_for_mask,
    estimate_target_quantiles,
    fit_scalers,
    write_evaluation_diagnostics,
)


def _single_sample_rank_fraction(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1)
    point_count = int(values.numel())
    ranks = torch.empty(point_count, dtype=torch.float32, device=values.device)
    if point_count <= 0:
        return ranks
    order = torch.argsort(torch.nan_to_num(values, nan=-torch.inf), descending=True, stable=True)
    ranks[order] = torch.arange(1, point_count + 1, dtype=torch.float32, device=values.device) / float(point_count)
    return ranks


def _false_peak_metrics(
    pred_log: torch.Tensor,
    target_raw: torch.Tensor,
    zero_threshold: float = 10.0,
    zero_cap: float = 10.0,
) -> dict[str, float]:
    pred_log = pred_log.reshape(-1)
    target_raw = target_raw.reshape(-1)
    if target_raw.numel() == 0:
        return {
            "false_peak_top5_rate": 0.0,
            "false_peak_top10_rate": 0.0,
            "hotspot_false_peak_margin_log": 0.0,
            "near_zero_points": 0.0,
            "near_zero_over_cap_rate": 0.0,
            "near_zero_over_log_mae": 0.0,
        }
    target_rank = _single_sample_rank_fraction(target_raw)
    pred_rank = _single_sample_rank_fraction(pred_log)
    false_background = target_rank >= 0.50
    pred_top5 = pred_rank <= 0.05
    pred_top10 = pred_rank <= 0.10
    top5_den = max(int(pred_top5.sum().item()), 1)
    top10_den = max(int(pred_top10.sum().item()), 1)
    hotspot = target_rank <= 0.05
    hard_negative = false_background & pred_top10
    if bool(hotspot.any()) and bool(hard_negative.any()):
        margin = pred_log[hotspot].median().item() - pred_log[hard_negative].median().item()
    else:
        margin = 0.0
    zero_mask = (target_raw < float(zero_threshold)) & false_background
    if bool(zero_mask.any()):
        cap_log = math.log1p(max(float(zero_cap), 0.0))
        over_delta = (pred_log[zero_mask] - cap_log).clamp_min(0.0)
        zero_points = int(zero_mask.sum().item())
        over_cap_rate = (over_delta > 0.0).to(torch.float32).mean().item()
        zero_over_log_mae = over_delta.mean().item()
    else:
        zero_points = 0
        over_cap_rate = 0.0
        zero_over_log_mae = 0.0
    return {
        "false_peak_top5_rate": (false_background & pred_top5).to(torch.float32).sum().item() / top5_den,
        "false_peak_top10_rate": (false_background & pred_top10).to(torch.float32).sum().item() / top10_den,
        "hotspot_false_peak_margin_log": margin,
        "near_zero_points": float(zero_points),
        "near_zero_over_cap_rate": over_cap_rate,
        "near_zero_over_log_mae": zero_over_log_mae,
    }


def build_graph_model(config: dict[str, Any], input_dim: int, edge_dim: int) -> NodeTransformerConv:
    model_cfg = dict(config.get("model", {}))
    return NodeTransformerConv(
        input_dim=input_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=1,
        num_layers=int(model_cfg.get("num_layers", 2)),
        heads=int(model_cfg.get("heads", 4)),
        edge_dim=edge_dim,
        dropout=float(model_cfg.get("dropout", 0.1)),
        activation=str(model_cfg.get("activation", "silu")),
        use_layer_norm=bool(model_cfg.get("layer_norm", True)),
        beta=bool(model_cfg.get("beta", True)),
    )


def train_graph_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[GraphBatch],
    optimizer: torch.optim.Optimizer,
    y_scaler: StandardScaler,
    device: torch.device,
    grad_clip: float,
    logger: Any | None = None,
    epoch: int | None = None,
    progress_every_steps: int = 50,
) -> dict[str, float]:
    model.train()
    totals = _empty_point_metric_totals()
    started_at = time.monotonic()
    epoch_label = f"{epoch:04d}" if epoch is not None else "????"
    total_loader_steps = len(loader)

    for loader_step, host_batch in enumerate(loader, start=1):
        if host_batch.num_points <= 0:
            continue
        batch = host_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch.features, batch.edge_index, batch.edge_attr)
        loss_values = F.smooth_l1_loss(prediction, batch.target_scaled, reduction="none").squeeze(-1)
        loss = (loss_values * batch.point_weights).sum() / batch.point_weights.sum().clamp_min(1e-12)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        pred_log, pred_raw = _decode_prediction(prediction.detach(), y_scaler)
        _update_point_metric_totals(
            totals,
            loss_sum=loss_values.sum().item(),
            weighted_loss_sum=(loss_values * batch.point_weights).sum().item(),
            weight_sum=batch.point_weights.sum().item(),
            pred_log=pred_log,
            pred_raw=pred_raw,
            target_log=batch.target_log,
            target_raw=batch.target_raw,
        )

        if logger is not None and (
            loader_step == 1
            or loader_step == total_loader_steps
            or (progress_every_steps > 0 and loader_step % progress_every_steps == 0)
        ):
            elapsed = time.monotonic() - started_at
            running_metrics = _finalize_point_metrics(totals)
            logger.info(
                (
                    "Epoch %s train progress | graph_batch=%s/%s | points=%s | edges=%s | elapsed=%s | "
                    "loss=%.6f | weighted_loss=%.6f | mae=%.6g | log_mae=%.6f | rel_mae=%.6f"
                ),
                epoch_label,
                loader_step,
                total_loader_steps,
                int(running_metrics["points"]),
                host_batch.num_edges,
                _format_duration(elapsed),
                running_metrics["loss"],
                running_metrics["weighted_loss"],
                running_metrics["earpiece_stress_mae"],
                running_metrics["earpiece_stress_log_mae"],
                running_metrics["earpiece_stress_relative_mae"],
            )
    return _finalize_point_metrics(totals)


def evaluate_graph(
    model: torch.nn.Module,
    loader: DataLoader[GraphBatch],
    y_scaler: StandardScaler,
    device: torch.device,
    split_name: str,
    epoch: int | None = None,
    logger: Any | None = None,
    progress_every_steps: int = 50,
    collect_diagnostics: bool = False,
) -> EvaluationResult:
    model.eval()
    totals = _empty_point_metric_totals()
    topk_totals = _empty_topk_metric_totals()
    top1_abs_sum = 0.0
    top1_log_abs_sum = 0.0
    top1_count = 0
    top5_abs_sum = 0.0
    top5_log_abs_sum = 0.0
    top5_count = 0
    peak_relative_error_sum = 0.0
    false_peak_top5_sum = 0.0
    false_peak_top10_sum = 0.0
    hotspot_false_peak_margin_sum = 0.0
    near_zero_points_sum = 0.0
    near_zero_over_cap_rate_sum = 0.0
    near_zero_over_log_mae_sum = 0.0
    sample_count = 0
    diagnostics: list[dict[str, Any]] = []
    started_at = time.monotonic()
    total_loader_steps = len(loader)
    epoch_label = f"{epoch:04d}" if epoch is not None else "????"

    with torch.no_grad():
        for loader_step, host_batch in enumerate(loader, start=1):
            if host_batch.num_points <= 0:
                continue
            batch = host_batch.to(device)
            prediction_scaled = model(batch.features, batch.edge_index, batch.edge_attr)
            loss_values = F.smooth_l1_loss(prediction_scaled, batch.target_scaled, reduction="none").squeeze(-1)
            pred_log, pred_raw = _decode_prediction(prediction_scaled, y_scaler)

            pred_log_cpu = pred_log.cpu()
            pred_raw_cpu = pred_raw.cpu()
            prediction_scaled_cpu = prediction_scaled.cpu()
            target_scaled_cpu = host_batch.target_scaled
            target_log_cpu = host_batch.target_log.squeeze(-1)
            target_raw_cpu = host_batch.target_raw
            graph_index_cpu = host_batch.graph_index
            weights_cpu = host_batch.point_weights

            abs_error = (pred_raw_cpu - target_raw_cpu).abs()
            log_abs_error = (pred_log_cpu - target_log_cpu).abs()
            _update_point_metric_totals(
                totals,
                loss_sum=loss_values.sum().item(),
                weighted_loss_sum=(loss_values.detach().cpu() * weights_cpu).sum().item(),
                weight_sum=weights_cpu.sum().item(),
                pred_log=pred_log_cpu,
                pred_raw=pred_raw_cpu,
                target_log=target_log_cpu,
                target_raw=target_raw_cpu,
            )

            for sample_idx in range(len(host_batch.names)):
                mask = graph_index_cpu == sample_idx
                if not bool(mask.any()):
                    continue
                sample_target = target_raw_cpu[mask]
                sample_pred = pred_raw_cpu[mask]
                sample_target_log = target_log_cpu[mask]
                sample_pred_log = pred_log_cpu[mask]
                sample_target_scaled = target_scaled_cpu[mask]
                sample_pred_scaled = prediction_scaled_cpu[mask]
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
                sample_v7_metrics = _false_peak_metrics(sample_pred_log, sample_target)
                false_peak_top5_sum += sample_v7_metrics["false_peak_top5_rate"]
                false_peak_top10_sum += sample_v7_metrics["false_peak_top10_rate"]
                hotspot_false_peak_margin_sum += sample_v7_metrics["hotspot_false_peak_margin_log"]
                near_zero_points_sum += sample_v7_metrics["near_zero_points"]
                near_zero_over_cap_rate_sum += sample_v7_metrics["near_zero_over_cap_rate"]
                near_zero_over_log_mae_sum += sample_v7_metrics["near_zero_over_log_mae"]

                top1_threshold = torch.quantile(sample_target, 0.99)
                top1_mask = sample_target >= top1_threshold
                top1_abs_sum += sample_abs[top1_mask].sum().item()
                sample_log_abs = log_abs_error[mask]
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
                sample_topk_masks = _rank_fraction_masks(sample_target)
                _update_topk_metric_totals(
                    topk_totals,
                    pred_log=sample_pred_log,
                    pred_raw=sample_pred,
                    target_log=sample_target_log,
                    target_raw=sample_target,
                )

                if collect_diagnostics:
                    sample_points = int(sample_target.numel())
                    sample_loss = F.smooth_l1_loss(
                        sample_pred_scaled,
                        sample_target_scaled,
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
                            "log_mae": sample_log_delta.abs().mean().item(),
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
                                sample_pred,
                                sample_target,
                                sample_topk_masks.get(
                                    "top1", torch.zeros_like(sample_target, dtype=torch.bool)
                                ),
                            ),
                            "top1_5_within25_ratio": _within25_ratio_for_mask(
                                sample_pred,
                                sample_target,
                                sample_topk_masks.get(
                                    "top1_5", torch.zeros_like(sample_target, dtype=torch.bool)
                                ),
                            ),
                            "top5_within25_ratio": _within25_ratio_for_mask(
                                sample_pred,
                                sample_target,
                                sample_topk_masks.get(
                                    "top5", torch.zeros_like(sample_target, dtype=torch.bool)
                                ),
                            ),
                            "top5_10_within25_ratio": _within25_ratio_for_mask(
                                sample_pred,
                                sample_target,
                                sample_topk_masks.get(
                                    "top5_10", torch.zeros_like(sample_target, dtype=torch.bool)
                                ),
                            ),
                            "top10_within25_ratio": _within25_ratio_for_mask(
                                sample_pred,
                                sample_target,
                                sample_topk_masks.get(
                                    "top10", torch.zeros_like(sample_target, dtype=torch.bool)
                                ),
                            ),
                            "under_pred_ratio": (sample_pred < sample_target).to(torch.float32).mean().item(),
                            "over_pred_ratio": (sample_pred > sample_target).to(torch.float32).mean().item(),
                            **sample_v7_metrics,
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
                        "Epoch %s %s progress | graph_batch=%s/%s | points=%s | edges=%s | elapsed=%s | "
                        "loss=%.6f | weighted_loss=%.6f | mae=%.6g | log_mae=%.6f | rel_mae=%.6f"
                    ),
                    epoch_label,
                    split_name,
                    loader_step,
                    total_loader_steps,
                    int(running_metrics["points"]),
                    host_batch.num_edges,
                    _format_duration(time.monotonic() - started_at),
                    running_metrics["loss"],
                    running_metrics["weighted_loss"],
                    running_metrics["earpiece_stress_mae"],
                    running_metrics["earpiece_stress_log_mae"],
                    running_metrics["earpiece_stress_relative_mae"],
                )

    metrics = _finalize_point_metrics(totals)
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
            "earpiece_stress_false_peak_top5_rate": false_peak_top5_sum / max(sample_count, 1),
            "earpiece_stress_false_peak_top10_rate": false_peak_top10_sum / max(sample_count, 1),
            "earpiece_stress_hotspot_false_peak_margin_log": hotspot_false_peak_margin_sum / max(sample_count, 1),
            "earpiece_stress_near_zero_points": near_zero_points_sum,
            "earpiece_stress_near_zero_over_cap_rate": near_zero_over_cap_rate_sum / max(sample_count, 1),
            "earpiece_stress_near_zero_over_log_mae": near_zero_over_log_mae_sum / max(sample_count, 1),
        }
    )
    metrics.update(_finalize_topk_metric_totals(topk_totals))
    return EvaluationResult(metrics=metrics, diagnostics=diagnostics)


class NodeTransformerConvTrainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.dataset_cfg = dict(config["dataset"])
        self.feature_cfg = dict(config.get("features", {}))
        self.graph_cfg = dict(config.get("graph", {}))
        self.training_cfg = dict(config["training"])
        self.scaler_cfg = dict(config.get("scaler", {}))
        self.target_cfg = dict(config.get("target", {}))
        self.loss_cfg = dict(config.get("loss", {}))

        self.save_dir = ensure_dir(self.training_cfg["save_dir"])
        self.logger = make_logger(self.save_dir, logger_name="case7_node_gnn")
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

    def _load_valid_scaler_cache(
        self,
        scaler_cache_path: str | Path | None,
        current_schema: dict[str, Any],
    ) -> tuple[StandardScaler, StandardScaler, dict[str, Any], float] | None:
        if scaler_cache_path is None or not Path(scaler_cache_path).exists():
            return None
        self.logger.info("Loading scaler cache: %s", scaler_cache_path)
        x_scaler, y_scaler, feature_schema = _load_scaler_cache(Path(scaler_cache_path))
        target_zero_threshold = _resolve_threshold_from_config(self.target_cfg, feature_schema, "zero_below")
        if not _scaler_cache_matches_schema(
            x_scaler,
            feature_schema,
            current_schema,
            target_zero_threshold=target_zero_threshold,
        ):
            self.logger.warning("Scaler cache feature schema mismatch; recomputing scalers.")
            return None
        feature_schema["target_zero_threshold"] = float(target_zero_threshold)
        return x_scaler, y_scaler, feature_schema, target_zero_threshold

    def _prepare(self) -> None:
        _require_transformer_conv()
        self.case_index = discover_case_index(self.dataset_cfg["root"])
        self.split_names = resolve_case_splits(self.dataset_cfg["root"], self.dataset_cfg)
        train_case_dirs = [self.case_index[name] for name in self.split_names["train"]]
        val_case_dirs = [self.case_index[name] for name in self.split_names["val"]]
        test_case_dirs = [self.case_index[name] for name in self.split_names.get("test", [])]

        self.train_sample_paths = expand_case_sample_paths(train_case_dirs, self.dataset_cfg)
        self.val_sample_paths = expand_case_sample_paths(val_case_dirs, self.dataset_cfg)
        self.test_sample_paths = expand_case_sample_paths(test_case_dirs, self.dataset_cfg)

        resolved_dataset = dict(self.dataset_cfg)
        resolved_dataset["split_mode"] = "explicit"
        resolved_dataset["train_cases"] = list(self.split_names["train"])
        resolved_dataset["val_cases"] = list(self.split_names["val"])
        resolved_dataset["test_cases"] = list(self.split_names.get("test", []))
        self.resolved_config = dict(self.config)
        self.resolved_config["dataset"] = resolved_dataset

        current_schema = _probe_current_feature_schema(self.train_sample_paths, self.dataset_cfg, self.feature_cfg)
        scaler_cache_path = self.scaler_cfg.get("cache_path")
        cache_payload = self._load_valid_scaler_cache(scaler_cache_path, current_schema=current_schema)
        target_zero_threshold = 0.0
        if cache_payload is not None:
            self.x_scaler, self.y_scaler, self.feature_schema, target_zero_threshold = cache_payload
        else:
            target_stats = estimate_target_quantiles(
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
                self.logger.info("Saving scaler cache: %s", scaler_cache_path)
                _save_scaler_cache(
                    Path(scaler_cache_path),
                    x_scaler=self.x_scaler,
                    y_scaler=self.y_scaler,
                    feature_schema=self.feature_schema,
                    config=self.resolved_config,
                )

        edge_attr_names = list(self.graph_cfg.get("edge_attr", ["dx", "dy", "dz", "dist"]))
        graph_schema = {
            "model_family": "transformerconv",
            "graph_edge_attr_names": edge_attr_names,
            "graph_edge_dim": len(edge_attr_names),
            "graph_add_reverse": bool(self.graph_cfg.get("add_reverse", True)),
            "graph_add_self_loops": bool(self.graph_cfg.get("add_self_loops", True)),
            "graph_standardize_edge_attr_per_graph": bool(self.graph_cfg.get("standardize_per_graph", True)),
            "target_zero_threshold": float(target_zero_threshold),
        }
        self.feature_schema.update(graph_schema)

        input_dim = int(self.feature_schema["input_dim"])
        edge_dim = int(self.feature_schema["graph_edge_dim"])
        self.model = build_graph_model(self.config, input_dim=input_dim, edge_dim=edge_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg.get("lr", 5e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 1e-4)),
        )

        write_yaml(self.save_dir / "resolved_config.yaml", self.resolved_config)
        write_json(self.save_dir / "feature_schema.json", self.feature_schema)

    def _make_loader(self, sample_paths: list[Path], shuffle: bool) -> DataLoader[GraphBatch]:
        assert self.x_scaler is not None and self.y_scaler is not None
        return make_graph_loader(
            sample_paths=sample_paths,
            dataset_cfg=self.dataset_cfg,
            feature_cfg=self.feature_cfg,
            graph_cfg=self.graph_cfg,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            feature_schema=self.feature_schema,
            target_cfg=self.target_cfg,
            loss_cfg=self.loss_cfg,
            graph_batch_size=int(self.training_cfg.get("graph_batch_size", 1)),
            num_workers=int(self.training_cfg.get("num_workers", 0)),
            shuffle=shuffle,
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
            "Train/Val/Test graph samples: %s/%s/%s",
            len(self.train_sample_paths),
            len(self.val_sample_paths),
            len(self.test_sample_paths),
        )
        self.logger.info("Feature schema: %s", json.dumps(self.feature_schema, ensure_ascii=False))
        self.logger.info("Target config: %s", json.dumps(self.target_cfg, ensure_ascii=False))
        self.logger.info("Loss config: %s", json.dumps(self.loss_cfg, ensure_ascii=False))
        self.logger.info("Graph config: %s", json.dumps(self.graph_cfg, ensure_ascii=False))
        self.logger.info("Training DataLoader workers: %s", int(self.training_cfg.get("num_workers", 0)))

        selection_metric = str(self.training_cfg.get("selection_metric", "earpiece_stress_log_mae"))
        best_score = float("inf")
        best_payload: dict[str, Any] | None = None
        wait = 0
        patience = int(self.training_cfg.get("early_stopping_patience", 8))
        grad_clip = float(self.training_cfg.get("grad_clip", 1.0))
        eval_every = int(self.training_cfg.get("eval_every", 1))
        progress_every_steps = int(self.training_cfg.get("progress_every_steps", 50))
        eval_progress_every_steps = int(self.training_cfg.get("eval_progress_every_steps", progress_every_steps))
        write_diagnostics = bool(self.training_cfg.get("write_diagnostics", True))

        for epoch in range(1, int(self.training_cfg.get("epochs", 30)) + 1):
            train_loader = self._make_loader(
                self.train_sample_paths,
                shuffle=bool(self.training_cfg.get("shuffle_samples", False)),
            )
            train_metrics = train_graph_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                y_scaler=self.y_scaler,
                device=self.device,
                grad_clip=grad_clip,
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

            val_loader = self._make_loader(self.val_sample_paths, shuffle=False)
            val_result = evaluate_graph(
                model=self.model,
                loader=val_loader,
                y_scaler=self.y_scaler,
                device=self.device,
                split_name="val",
                epoch=epoch,
                logger=self.logger,
                progress_every_steps=eval_progress_every_steps,
                collect_diagnostics=write_diagnostics,
            )
            val_metrics = val_result.metrics
            history_row["val_loss"] = round(float(val_metrics["loss"]), 8)
            for key, value in val_metrics.items():
                if key not in {"loss", "weighted_loss"}:
                    history_row[f"val_{key}"] = round(float(value), 8)
            history_row["val_weighted_loss"] = round(float(val_metrics.get("weighted_loss", val_metrics["loss"])), 8)
            self._append_history_row(history_row)
            self.logger.info(
                "Epoch %04d | train=%s | val=%s",
                epoch,
                json.dumps(train_metrics, ensure_ascii=False),
                json.dumps(val_metrics, ensure_ascii=False),
            )

            if selection_metric not in val_metrics:
                raise KeyError(f"Selection metric '{selection_metric}' is unavailable: {sorted(val_metrics)}")
            score = float(val_metrics[selection_metric])
            if score < best_score:
                best_score = score
                wait = 0
                test_metrics: dict[str, float] = {}
                test_result = EvaluationResult(metrics={}, diagnostics=[])
                if self.test_sample_paths:
                    test_loader = self._make_loader(self.test_sample_paths, shuffle=False)
                    test_result = evaluate_graph(
                        model=self.model,
                        loader=test_loader,
                        y_scaler=self.y_scaler,
                        device=self.device,
                        split_name="test",
                        epoch=epoch,
                        logger=self.logger,
                        progress_every_steps=eval_progress_every_steps,
                        collect_diagnostics=write_diagnostics,
                    )
                    test_metrics = test_result.metrics
                if write_diagnostics:
                    write_evaluation_diagnostics(self.save_dir / "best_val_diagnostics.csv", val_result.diagnostics)
                    if self.test_sample_paths:
                        write_evaluation_diagnostics(self.save_dir / "best_test_diagnostics.csv", test_result.diagnostics)
                best_payload = {
                    "epoch": epoch,
                    "selection_metric": selection_metric,
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
        self.logger.info("Best run summary:\n%s", json.dumps(best_payload, indent=2, ensure_ascii=False))
        return best_payload

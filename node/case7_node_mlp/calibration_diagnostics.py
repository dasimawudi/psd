from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.hotspot_within25_diagnostics import GroupStats, _load_model
from case7_node_mlp.models import regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, make_loader
from case7_node_mlp.within25_diagnostics import (
    FREQUENCY_BINS,
    MODE_PROXIMITY_LABELS,
    REGION_LABELS,
    _bin_indices,
    _case_mode_frequencies,
    _case_region_arrays,
    _exclusive_region_ids,
    _labels,
    _mode_proximity_id,
)


PRED_MAG_BINS = [-float("inf"), 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, float("inf")]
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50)


@dataclass
class PredictionTable:
    target_raw: np.ndarray
    pred_raw: np.ndarray
    target_log: np.ndarray
    pred_log: np.ndarray
    frequency_ids: np.ndarray
    region_ids: np.ndarray
    mode_ids: np.ndarray
    pred_mag_ids: np.ndarray
    rank_fraction: np.ndarray
    hotspot_union_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc log-bias calibration diagnostics.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--fit-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--min-bucket-count", type=int, default=5000)
    parser.add_argument("--clip-bias", type=float, default=1.0, help="Clip learned log bias to +/- this value.")
    return parser.parse_args()


def _format_edge(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def _pred_mag_labels() -> list[str]:
    labels = []
    for left, right in zip(PRED_MAG_BINS[:-1], PRED_MAG_BINS[1:]):
        labels.append(f"[{_format_edge(left)}, {_format_edge(right)})")
    return labels


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1, dtype=np.int64)
    return ranks.astype(np.float32) / float(n)


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros(values.shape[0], dtype=bool)
    if values.size == 0:
        return mask
    count = max(1, int(math.ceil(values.shape[0] * fraction)))
    if count >= values.size:
        mask[:] = True
        return mask
    top_indices = np.argpartition(np.nan_to_num(values, nan=-np.inf), -count)[-count:]
    mask[top_indices] = True
    return mask


def _make_loader_for_split(
    *,
    split: str,
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
) -> tuple[Any, dict[str, Path]]:
    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[split]]
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
    return loader, case_index


def _collect_predictions(
    *,
    split: str,
    loader: Any,
    case_index: dict[str, Path],
    model: torch.nn.Module,
    y_scaler: StandardScaler,
    device: torch.device,
    point_batch_size: int,
    logger: Any,
) -> PredictionTable:
    target_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    target_log_parts: list[np.ndarray] = []
    pred_log_parts: list[np.ndarray] = []
    frequency_id_parts: list[np.ndarray] = []
    region_id_parts: list[np.ndarray] = []
    mode_id_parts: list[np.ndarray] = []
    pred_mag_id_parts: list[np.ndarray] = []
    rank_fraction_parts: list[np.ndarray] = []
    case_parts: list[np.ndarray] = []
    node_parts: list[np.ndarray] = []
    sample_top_parts: list[np.ndarray] = []
    case_region_cache: dict[str, dict[str, np.ndarray]] = {}
    case_mode_cache: dict[str, np.ndarray] = {}
    case_name_to_id: dict[str, int] = {}
    case_hotspot_nodes: dict[int, set[int]] = defaultdict(set)
    frequency_labels = _labels(FREQUENCY_BINS)

    with torch.no_grad():
        for loader_step, host_batch in enumerate(loader, start=1):
            predictions_scaled = []
            for start in range(0, host_batch.num_points, point_batch_size):
                end = min(start + point_batch_size, host_batch.num_points)
                features = host_batch.features[start:end].to(device, non_blocking=True)
                predictions_scaled.append(regression_output(model(features)).detach().cpu())
            prediction_scaled = torch.cat(predictions_scaled, dim=0)
            pred_log_t, pred_raw_t = _decode_prediction(prediction_scaled.to(device), y_scaler)
            pred_log = pred_log_t.cpu().numpy().astype(np.float32, copy=False)
            pred_raw = pred_raw_t.cpu().numpy().astype(np.float32, copy=False)
            target_log = host_batch.target_log.squeeze(-1).numpy().astype(np.float32, copy=False)
            target_raw = host_batch.target_raw.numpy().astype(np.float32, copy=False)
            sample_index = host_batch.sample_index.numpy().astype(np.int64, copy=False)
            node_rows = host_batch.node_indices.numpy().astype(np.int64, copy=False)

            frequency_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            region_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            mode_ids = np.zeros(target_raw.shape[0], dtype=np.int16)
            rank_fraction = np.ones(target_raw.shape[0], dtype=np.float32)
            case_ids = np.zeros(target_raw.shape[0], dtype=np.int32)
            sample_top = np.zeros(target_raw.shape[0], dtype=bool)

            for sample_idx, case_name in enumerate(host_batch.case_names):
                sample_mask = sample_index == sample_idx
                if not sample_mask.any():
                    continue
                if case_name not in case_name_to_id:
                    case_name_to_id[case_name] = len(case_name_to_id)
                case_id = case_name_to_id[case_name]
                selected_positions = np.flatnonzero(sample_mask)
                case_ids[sample_mask] = case_id
                sample_frequency = float(host_batch.frequency_hz[sample_idx].item())
                frequency_ids[sample_mask] = _bin_indices(np.array([sample_frequency], dtype=np.float32), FREQUENCY_BINS)[0]
                if case_name not in case_region_cache:
                    case_region_cache[case_name] = _case_region_arrays(case_index[case_name])
                region_ids[sample_mask] = _exclusive_region_ids(case_region_cache[case_name], node_rows[sample_mask])
                if case_name not in case_mode_cache:
                    case_mode_cache[case_name] = _case_mode_frequencies(case_index[case_name])
                mode_ids[sample_mask] = _mode_proximity_id(case_mode_cache[case_name], sample_frequency)
                local_rank = _rank_fraction(target_raw[sample_mask])
                rank_fraction[sample_mask] = local_rank
                local_top = _top_fraction_mask(target_raw[sample_mask], 0.05)
                sample_top[selected_positions[local_top]] = True
                case_hotspot_nodes[case_id].update(node_rows[selected_positions[local_top]].tolist())

            pred_mag_ids = _bin_indices(pred_log, PRED_MAG_BINS).astype(np.int16, copy=False)
            target_parts.append(target_raw.copy())
            pred_parts.append(pred_raw.copy())
            target_log_parts.append(target_log.copy())
            pred_log_parts.append(pred_log.copy())
            frequency_id_parts.append(frequency_ids.copy())
            region_id_parts.append(region_ids.copy())
            mode_id_parts.append(mode_ids.copy())
            pred_mag_id_parts.append(pred_mag_ids.copy())
            rank_fraction_parts.append(rank_fraction.copy())
            case_parts.append(case_ids.copy())
            node_parts.append(node_rows.copy())
            sample_top_parts.append(sample_top.copy())

            if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
                logger.info(
                    "collect %s | sample_batch=%s/%s | points=%s | frequency_buckets=%s",
                    split,
                    loader_step,
                    len(loader),
                    sum(part.size for part in target_parts),
                    len(frequency_labels),
                )

    target_raw = np.concatenate(target_parts)
    pred_raw = np.concatenate(pred_parts)
    target_log = np.concatenate(target_log_parts)
    pred_log = np.concatenate(pred_log_parts)
    frequency_ids = np.concatenate(frequency_id_parts)
    region_ids = np.concatenate(region_id_parts)
    mode_ids = np.concatenate(mode_id_parts)
    pred_mag_ids = np.concatenate(pred_mag_id_parts)
    rank_fraction = np.concatenate(rank_fraction_parts)
    case_ids_all = np.concatenate(case_parts)
    node_rows_all = np.concatenate(node_parts)
    hotspot_union_mask = np.zeros(target_raw.shape[0], dtype=bool)
    for case_id, nodes in case_hotspot_nodes.items():
        case_mask = case_ids_all == case_id
        if case_mask.any():
            hotspot_union_mask[case_mask] = np.isin(node_rows_all[case_mask], list(nodes))
    return PredictionTable(
        target_raw=target_raw,
        pred_raw=pred_raw,
        target_log=target_log,
        pred_log=pred_log,
        frequency_ids=frequency_ids,
        region_ids=region_ids,
        mode_ids=mode_ids,
        pred_mag_ids=pred_mag_ids,
        rank_fraction=rank_fraction,
        hotspot_union_mask=hotspot_union_mask,
    )


def _bucket_key(table: PredictionTable, scheme: str) -> np.ndarray:
    n = table.target_raw.shape[0]
    if scheme in {"baseline", "global"}:
        return np.zeros(n, dtype=np.int64)
    if scheme == "pred_mag":
        return table.pred_mag_ids.astype(np.int64)
    if scheme == "frequency_pred_mag":
        return table.frequency_ids.astype(np.int64) * len(PRED_MAG_BINS) + table.pred_mag_ids.astype(np.int64)
    if scheme == "region_pred_mag":
        return table.region_ids.astype(np.int64) * len(PRED_MAG_BINS) + table.pred_mag_ids.astype(np.int64)
    if scheme == "mode_pred_mag":
        return table.mode_ids.astype(np.int64) * len(PRED_MAG_BINS) + table.pred_mag_ids.astype(np.int64)
    raise ValueError(f"Unknown calibration scheme: {scheme}")


def _fit_biases(
    table: PredictionTable,
    scheme: str,
    min_count: int,
    clip_bias: float,
) -> dict[str, Any]:
    if scheme == "baseline":
        return {
            "scheme": scheme,
            "global_bias": 0.0,
            "biases": {},
            "counts": {0: int(table.target_raw.shape[0])},
            "min_count": int(min_count),
            "clip_bias": float(clip_bias),
        }
    keys = _bucket_key(table, scheme)
    residual = table.target_log.astype(np.float64) - table.pred_log.astype(np.float64)
    global_bias = float(np.clip(np.mean(residual), -clip_bias, clip_bias))
    biases: dict[int, float] = {}
    counts: dict[int, int] = {}
    for key in np.unique(keys):
        mask = keys == key
        count = int(mask.sum())
        counts[int(key)] = count
        if count >= int(min_count):
            biases[int(key)] = float(np.clip(np.mean(residual[mask]), -clip_bias, clip_bias))
    return {
        "scheme": scheme,
        "global_bias": global_bias,
        "biases": biases,
        "counts": counts,
        "min_count": int(min_count),
        "clip_bias": float(clip_bias),
    }


def _apply_bias(table: PredictionTable, fit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keys = _bucket_key(table, str(fit["scheme"]))
    corrected_log = table.pred_log.astype(np.float32, copy=True)
    global_bias = float(fit["global_bias"])
    biases = {int(key): float(value) for key, value in fit["biases"].items()}
    bias_values = np.full(corrected_log.shape[0], global_bias, dtype=np.float32)
    for key, bias in biases.items():
        bias_values[keys == key] = bias
    corrected_log = corrected_log + bias_values
    corrected_raw = np.expm1(corrected_log).clip(min=0.0).astype(np.float32, copy=False)
    return corrected_log, corrected_raw


def _metric_row(table: PredictionTable, pred_log: np.ndarray, pred_raw: np.ndarray, group: str, mask: np.ndarray) -> dict[str, Any]:
    stats = GroupStats(relative_errors=[])
    stats.update(table.target_raw[mask], pred_raw[mask], table.target_log[mask], pred_log[mask])
    return stats.row(group)


def _evaluate_predictions(table: PredictionTable, pred_log: np.ndarray, pred_raw: np.ndarray, scheme: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_mask = np.ones(table.target_raw.shape[0], dtype=bool)
    rows.append({"section": "overall", **_metric_row(table, pred_log, pred_raw, "overall", all_mask), "scheme": scheme})
    for fraction in TOP_FRACTIONS:
        label = f"top{int(round(100 * fraction))}"
        rows.append({"section": "cumulative_topk", **_metric_row(table, pred_log, pred_raw, label, table.rank_fraction <= fraction), "scheme": scheme})
    band_defs = [(0.0, 0.01, "top1"), (0.01, 0.05, "1-5"), (0.05, 0.10, "5-10"), (0.10, 0.25, "10-25"), (0.25, 0.50, "25-50"), (0.50, 1.0, "50-100")]
    for left, right, label in band_defs:
        mask = (table.rank_fraction > left) & (table.rank_fraction <= right)
        rows.append({"section": "topk_band", **_metric_row(table, pred_log, pred_raw, label, mask), "scheme": scheme})
    rows.append({"section": "hotspot_union", **_metric_row(table, pred_log, pred_raw, "hotspot_top5_union", table.hotspot_union_mask), "scheme": scheme})
    rows.append({"section": "hotspot_union", **_metric_row(table, pred_log, pred_raw, "non_hotspot", ~table.hotspot_union_mask), "scheme": scheme})
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scheme", "section", "group", *[key for key in rows[0] if key not in {"scheme", "section", "group"}]]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_scheme_comparison(path: Path, rows: list[dict[str, Any]], section: str, groups: list[str], title: str) -> None:
    df = pd.DataFrame(rows)
    df = df[(df["section"] == section) & (df["group"].isin(groups))]
    if df.empty:
        return
    pivot = df.pivot(index="scheme", columns="group", values="within25_ratio").reindex(columns=groups)
    ax = (pivot * 100.0).plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.axhline(90.0, color="#222222", linestyle="--", linewidth=1.1)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("within25_ratio (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bias_summary_rows(fits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scheme, fit in fits.items():
        values = np.array(list(fit["biases"].values()), dtype=np.float64)
        counts = np.array(list(fit["counts"].values()), dtype=np.int64)
        rows.append(
            {
                "scheme": scheme,
                "global_bias": fit["global_bias"],
                "fit_buckets": int(values.size),
                "all_buckets": int(counts.size),
                "bias_mean": float(values.mean()) if values.size else float("nan"),
                "bias_min": float(values.min()) if values.size else float("nan"),
                "bias_max": float(values.max()) if values.size else float("nan"),
                "bucket_count_min": int(counts.min()) if counts.size else 0,
                "bucket_count_max": int(counts.max()) if counts.size else 0,
            }
        )
    return rows


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

    if args.max_cases is not None:
        dataset_cfg["max_cases"] = int(args.max_cases)
        for split in ("train", "val", "test"):
            if dataset_cfg.get("split_mode") == "explicit":
                dataset_cfg[split + "_cases"] = list(dataset_cfg.get(split + "_cases", []))[: int(args.max_cases)]
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"calibration_{args.fit_split}_to_{args.eval_split}")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.calibration_diagnostics", log_file="calibration_diagnostics.log")
    device = resolve_device(args.device)
    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = _load_model(checkpoint, config, device)
    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))

    fit_loader, fit_case_index = _make_loader_for_split(
        split=args.fit_split,
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
    eval_loader, eval_case_index = _make_loader_for_split(
        split=args.eval_split,
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
    fit_table = _collect_predictions(
        split=args.fit_split,
        loader=fit_loader,
        case_index=fit_case_index,
        model=model,
        y_scaler=y_scaler,
        device=device,
        point_batch_size=point_batch_size,
        logger=logger,
    )
    eval_table = _collect_predictions(
        split=args.eval_split,
        loader=eval_loader,
        case_index=eval_case_index,
        model=model,
        y_scaler=y_scaler,
        device=device,
        point_batch_size=point_batch_size,
        logger=logger,
    )

    schemes = ["baseline", "global", "pred_mag", "frequency_pred_mag", "region_pred_mag", "mode_pred_mag"]
    fits: dict[str, dict[str, Any]] = {
        scheme: _fit_biases(fit_table, scheme, int(args.min_bucket_count), float(args.clip_bias)) for scheme in schemes
    }
    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        pred_log, pred_raw = _apply_bias(eval_table, fits[scheme])
        rows.extend(_evaluate_predictions(eval_table, pred_log, pred_raw, scheme))

    metrics_csv = output_dir / "calibration_test_metrics.csv"
    _write_rows(metrics_csv, rows)
    bias_rows = _bias_summary_rows(fits)
    bias_csv = output_dir / "calibration_bias_summary.csv"
    _write_rows(bias_csv, [{"scheme": row.pop("scheme"), "section": "bias", "group": "fit", **row} for row in bias_rows])

    _plot_scheme_comparison(
        output_dir / "calibration_overall_bottom_hotspot.png",
        rows,
        "topk_band",
        ["top1", "1-5", "5-10", "10-25", "25-50", "50-100"],
        "Calibration effect by true target band",
    )
    _plot_scheme_comparison(
        output_dir / "calibration_cumulative_topk.png",
        rows,
        "cumulative_topk",
        ["top1", "top5", "top10", "top25", "top50"],
        "Calibration effect on cumulative topK",
    )
    _plot_scheme_comparison(
        output_dir / "calibration_hotspot_union.png",
        rows,
        "hotspot_union",
        ["hotspot_top5_union", "non_hotspot"],
        "Calibration effect on hotspot vs non-hotspot",
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "schemes": schemes,
        "min_bucket_count": int(args.min_bucket_count),
        "clip_bias": float(args.clip_bias),
        "pred_mag_bins": _pred_mag_labels(),
        "frequency_bins": _labels(FREQUENCY_BINS),
        "region_labels": REGION_LABELS,
        "mode_labels": MODE_PROXIMITY_LABELS,
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "bias_summary_csv": str(bias_csv),
            "band_plot": str(output_dir / "calibration_overall_bottom_hotspot.png"),
            "cumulative_topk_plot": str(output_dir / "calibration_cumulative_topk.png"),
            "hotspot_union_plot": str(output_dir / "calibration_hotspot_union.png"),
        },
    }
    write_json(output_dir / "calibration_diagnostics_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import PointMLP, regression_output
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, build_model, make_loader


@dataclass
class GroupStats:
    count: int = 0
    relative_count: int = 0
    within25_count: int = 0
    abs_sum: float = 0.0
    log_abs_sum: float = 0.0
    relative_sum: float = 0.0
    symmetric_relative_sum: float = 0.0
    under_count: int = 0
    over_count: int = 0
    target_sum: float = 0.0
    pred_sum: float = 0.0
    relative_errors: list[np.ndarray] | None = None

    def update(self, target_raw: np.ndarray, pred_raw: np.ndarray, target_log: np.ndarray, pred_log: np.ndarray) -> None:
        if target_raw.size == 0:
            return
        delta = pred_raw - target_raw
        abs_error = np.abs(delta).astype(np.float64, copy=False)
        log_abs = np.abs(pred_log - target_log).astype(np.float64, copy=False)
        relative_mask = np.abs(target_raw) > 1e-12
        relative = abs_error[relative_mask] / np.abs(target_raw[relative_mask]).astype(np.float64, copy=False)
        symmetric = abs_error / np.maximum(0.5 * (np.abs(pred_raw) + np.abs(target_raw)), 1e-12)

        self.count += int(target_raw.size)
        self.relative_count += int(relative_mask.sum())
        self.within25_count += int((relative <= 0.25).sum())
        self.abs_sum += float(abs_error.sum())
        self.log_abs_sum += float(log_abs.sum())
        self.relative_sum += float(relative.sum())
        self.symmetric_relative_sum += float(symmetric.sum())
        self.under_count += int((pred_raw < target_raw).sum())
        self.over_count += int((pred_raw > target_raw).sum())
        self.target_sum += float(target_raw.sum())
        self.pred_sum += float(pred_raw.sum())
        if self.relative_errors is not None:
            self.relative_errors.append(relative.astype(np.float32, copy=False))

    def row(self, group: str) -> dict[str, Any]:
        points = max(self.count, 1)
        relative_points = max(self.relative_count, 1)
        if self.relative_errors:
            errors = np.concatenate(self.relative_errors)
            p50 = float(np.quantile(errors, 0.50)) if errors.size else 0.0
            p75 = float(np.quantile(errors, 0.75)) if errors.size else 0.0
            p90 = float(np.quantile(errors, 0.90)) if errors.size else 0.0
            p95 = float(np.quantile(errors, 0.95)) if errors.size else 0.0
        else:
            p50 = p75 = p90 = p95 = float("nan")
        within25 = self.within25_count / relative_points
        return {
            "group": group,
            "points": self.count,
            "relative_points": self.relative_count,
            "within25_ratio": within25,
            "miss25_rate": 1.0 - within25,
            "mae": self.abs_sum / points,
            "log_mae": self.log_abs_sum / points,
            "relative_mae": self.relative_sum / relative_points,
            "symmetric_relative_mae": self.symmetric_relative_sum / points,
            "relative_error_p50": p50,
            "relative_error_p75": p75,
            "relative_error_p90": p90,
            "relative_error_p95": p95,
            "under_pred_ratio": self.under_count / points,
            "over_pred_ratio": self.over_count / points,
            "target_mean": self.target_sum / points,
            "pred_mean": self.pred_sum / points,
            "pred_target_ratio": self.pred_sum / max(self.target_sum, 1e-12),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate within25 by true hotspot and non-hotspot nodes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument("--top-fraction", type=float, default=0.05, help="Per case-frequency true target top fraction.")
    parser.add_argument(
        "--node-union",
        action="store_true",
        help="Treat a node as hotspot for all frequencies if it is top-fraction in any frequency of the same case.",
    )
    return parser.parse_args()


def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    mask = np.zeros(values.shape[0], dtype=bool)
    if values.size == 0:
        return mask
    count = max(1, int(math.ceil(values.shape[0] * max(float(fraction), 0.0))))
    if count >= values.size:
        mask[:] = True
        return mask
    top_indices = np.argpartition(np.nan_to_num(values, nan=-np.inf), -count)[-count:]
    mask[top_indices] = True
    return mask


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_within25(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["group"]) for row in rows]
    values = [100.0 * float(row["within25_ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#d64f3a", "#3b7bbf", "#777777"][: len(labels)])
    ax.axhline(90.0, color="#222222", linestyle="--", linewidth=1.2, label="90% target")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("within25_ratio (%)")
    ax.set_title("Within25 by true hotspot group")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_error_cdf(path: Path, stats: dict[str, GroupStats]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, stat in stats.items():
        if not stat.relative_errors:
            continue
        errors = np.concatenate(stat.relative_errors)
        if errors.size == 0:
            continue
        clipped = np.sort(np.clip(errors, 0.0, 3.0))
        y = np.linspace(0.0, 1.0, clipped.size, endpoint=True)
        ax.plot(clipped, y, label=label)
    ax.axvline(0.25, color="#222222", linestyle="--", linewidth=1.2, label="25% threshold")
    ax.set_xlabel("relative error")
    ax.set_ylabel("CDF")
    ax.set_title("Relative Error CDF by True Hotspot Group")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_model(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device) -> PointMLP:
    feature_schema = dict(checkpoint["feature_schema"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"]), feature_schema=feature_schema).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _precompute_case_hotspot_nodes(loader: Any, top_fraction: float, logger: Any) -> dict[str, set[int]]:
    case_hotspot_nodes: dict[str, set[int]] = {}
    for loader_step, host_batch in enumerate(loader, start=1):
        target_raw = host_batch.target_raw.numpy().astype(np.float32, copy=False)
        sample_index = host_batch.sample_index.numpy().astype(np.int64, copy=False)
        node_rows = host_batch.node_indices.numpy().astype(np.int64, copy=False)
        for sample_idx, case_name in enumerate(host_batch.case_names):
            sample_mask = sample_index == sample_idx
            if not sample_mask.any():
                continue
            selected_positions = np.flatnonzero(sample_mask)
            local_top = _top_fraction_mask(target_raw[sample_mask], top_fraction)
            case_hotspot_nodes.setdefault(case_name, set()).update(node_rows[selected_positions[local_top]].tolist())
        if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
            logger.info(
                "precompute hotspot-node union | sample_batch=%s/%s | cases=%s | hotspot_nodes=%s",
                loader_step,
                len(loader),
                len(case_hotspot_nodes),
                sum(len(nodes) for nodes in case_hotspot_nodes.values()),
            )
    return case_hotspot_nodes


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
        if dataset_cfg.get("split_mode") == "explicit":
            dataset_cfg[args.split + "_cases"] = list(dataset_cfg.get(args.split + "_cases", []))[: int(args.max_cases)]
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"hotspot_within25_{args.split}")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.hotspot_within25", log_file="hotspot_within25.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = _load_model(checkpoint, config, device)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[args.split]]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Running hotspot within25 diagnostics | split=%s | cases=%s | samples=%s | top_fraction=%.4f | node_union=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        float(args.top_fraction),
        bool(args.node_union),
        device,
    )

    loader = make_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=int(args.sample_batch_size or training_cfg.get("sample_batch_size", 1)),
        num_workers=int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)),
        shuffle=False,
    )

    stats = {
        "hotspot_top5": GroupStats(relative_errors=[]),
        "non_hotspot": GroupStats(relative_errors=[]),
        "overall": GroupStats(relative_errors=[]),
    }
    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))
    case_hotspot_nodes: dict[str, set[int]] = {}
    if bool(args.node_union):
        logger.info("Precomputing case-level hotspot node union before model inference.")
        case_hotspot_nodes = _precompute_case_hotspot_nodes(loader, float(args.top_fraction), logger)
        logger.info(
            "Hotspot node union ready | cases=%s | hotspot_nodes=%s",
            len(case_hotspot_nodes),
            sum(len(nodes) for nodes in case_hotspot_nodes.values()),
        )

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

            hotspot_mask = np.zeros(target_raw.shape[0], dtype=bool)
            for sample_idx, case_name in enumerate(host_batch.case_names):
                sample_mask = sample_index == sample_idx
                if not sample_mask.any():
                    continue
                selected_positions = np.flatnonzero(sample_mask)
                if bool(args.node_union):
                    hotspot_nodes = case_hotspot_nodes.get(case_name, set())
                    hotspot_mask[selected_positions] = np.isin(node_rows[sample_mask], list(hotspot_nodes))
                    continue
                local_top = _top_fraction_mask(target_raw[sample_mask], float(args.top_fraction))
                hotspot_mask[selected_positions[local_top]] = True

            non_hotspot_mask = ~hotspot_mask
            stats["hotspot_top5"].update(target_raw[hotspot_mask], pred_raw[hotspot_mask], target_log[hotspot_mask], pred_log[hotspot_mask])
            stats["non_hotspot"].update(target_raw[non_hotspot_mask], pred_raw[non_hotspot_mask], target_log[non_hotspot_mask], pred_log[non_hotspot_mask])
            stats["overall"].update(target_raw, pred_raw, target_log, pred_log)

            if loader_step == 1 or loader_step == len(loader) or loader_step % 20 == 0:
                logger.info(
                    "progress | sample_batch=%s/%s | points=%s | hotspot_points=%s",
                    loader_step,
                    len(loader),
                    stats["overall"].count,
                    stats["hotspot_top5"].count,
                )

    rows = [stats["hotspot_top5"].row("hotspot_top5"), stats["non_hotspot"].row("non_hotspot"), stats["overall"].row("overall")]
    csv_path = output_dir / "hotspot_within25_summary.csv"
    _write_rows(csv_path, rows)
    plot_path = output_dir / "hotspot_within25.png"
    _plot_within25(plot_path, rows)
    cdf_path = output_dir / "hotspot_relative_error_cdf.png"
    _plot_error_cdf(cdf_path, stats)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "definition": {
            "hotspot": (
                f"case-level node union of per case-frequency true target top {100.0 * float(args.top_fraction):.2f}% nodes"
                if bool(args.node_union)
                else f"per case-frequency true target top {100.0 * float(args.top_fraction):.2f}% nodes"
            ),
            "node_union": bool(args.node_union),
        },
        "outputs": {
            "summary_csv": str(csv_path),
            "within25_plot": str(plot_path),
            "relative_error_cdf_plot": str(cdf_path),
        },
        "rows": rows,
    }
    write_json(output_dir / "hotspot_within25_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

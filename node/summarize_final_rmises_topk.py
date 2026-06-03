from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.evaluate import _load_checkpoint
from case7_node_mlp.models import PointMLP
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import _decode_prediction, make_loader


SUMMARY_BANDS = {
    "top0_1pct": 0.001,
    "top0_5pct": 0.005,
    "top1pct": 0.01,
    "top5pct": 0.05,
    "top10pct": 0.10,
}

SUMMARY_NODE_COUNTS = {
    "top1_nodes": 1,
    "top5_nodes": 5,
    "top10_nodes": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize final per-node RMises from predicted and true MISES PSD density."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to node MLP best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--point-batch-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-frames-per-case", type=int, default=None)
    parser.add_argument(
        "--write-per-node",
        action="store_true",
        help="Write per-node final RMises CSVs for every case. Top CSVs are always written.",
    )
    return parser.parse_args()


def _load_model(checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device) -> PointMLP:
    feature_schema = dict(checkpoint["feature_schema"])
    model = PointMLP(
        input_dim=int(feature_schema["input_dim"]),
        hidden_dims=[int(dim) for dim in config.get("model", {}).get("hidden_dims", [256, 256, 128])],
        dropout=float(config.get("model", {}).get("dropout", 0.0)),
        activation=str(config.get("model", {}).get("activation", "silu")),
        use_layer_norm=bool(config.get("model", {}).get("layer_norm", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _trapz_weights(frequencies: np.ndarray) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(frequencies)
    sorted_freq = frequencies[order]
    weights_sorted = np.zeros_like(sorted_freq, dtype=np.float64)
    if sorted_freq.size == 1:
        weights_sorted[0] = 0.0
    else:
        deltas = np.diff(sorted_freq)
        weights_sorted[0] = 0.5 * deltas[0]
        weights_sorted[-1] = 0.5 * deltas[-1]
        if sorted_freq.size > 2:
            weights_sorted[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    weights = np.empty_like(weights_sorted)
    weights[order] = weights_sorted
    return weights


def _read_node_table(case_dir: Path) -> pd.DataFrame:
    usecols = lambda column: column in {"node_index", "node_label", "x", "y", "z"}
    return pd.read_csv(case_dir / "nodes.csv", usecols=usecols)


def _merge_native_rmises(case_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    native_path = case_dir / "final_rmises.csv"
    if not native_path.exists():
        return rows
    native = pd.read_csv(native_path, usecols=lambda column: column in {"node_index", "RMises_native"})
    if "RMises_native" in native.columns:
        rows = rows.merge(native, on="node_index", how="left")
    return rows


def _safe_relative_error(pred: pd.Series, target: pd.Series) -> pd.Series:
    denom = target.abs()
    return (pred - target).abs() / denom.where(denom > 1e-12, np.nan)


def _summarize_group(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    count = int(len(df))
    if count == 0:
        return {"group": label, "nodes": 0}
    error = df["pred_rmises"] - df["target_rmises"]
    abs_error = error.abs()
    rel_error = _safe_relative_error(df["pred_rmises"], df["target_rmises"])
    summary: dict[str, float | str] = {
        "group": label,
        "nodes": count,
        "target_rmises_mean": float(df["target_rmises"].mean()),
        "target_rmises_max": float(df["target_rmises"].max()),
        "pred_rmises_mean": float(df["pred_rmises"].mean()),
        "pred_rmises_max": float(df["pred_rmises"].max()),
        "mae": float(abs_error.mean()),
        "rmse": float(math.sqrt(float((error.pow(2)).mean()))),
        "bias": float(error.mean()),
        "relative_mae": float(rel_error.mean(skipna=True)),
        "within1_ratio": float((rel_error <= 0.01).mean()),
        "within5_ratio": float((rel_error <= 0.05).mean()),
        "within10_ratio": float((rel_error <= 0.10).mean()),
        "within15_ratio": float((rel_error <= 0.15).mean()),
        "within20_ratio": float((rel_error <= 0.20).mean()),
        "within25_ratio": float((rel_error <= 0.25).mean()),
        "pred_target_mean_ratio": float(df["pred_rmises"].mean() / max(abs(float(df["target_rmises"].mean())), 1e-12)),
    }
    if "RMises_native" in df.columns:
        native_rel = _safe_relative_error(df["target_rmises"], df["RMises_native"])
        summary["native_check_relative_mae"] = float(native_rel.mean(skipna=True))
    return summary


def _top_count(total: int, fraction: float) -> int:
    return min(total, max(1, int(math.ceil(float(total) * float(fraction)))))


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

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"final_rmises_topk_{args.split}")
    cases_dir = ensure_dir(output_dir / "cases")
    logger = make_logger(output_dir, logger_name="case7_node_mlp.final_rmises_topk", log_file="final_rmises_topk.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = _load_model(checkpoint=checkpoint, config=config, device=device)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_names = list(split_names[args.split])
    case_dirs = [case_index[name] for name in case_names]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Summarizing final RMises | split=%s | cases=%s | samples=%s | checkpoint=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        checkpoint_path,
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

    case_accum: dict[str, dict[str, Any]] = {}
    sample_count = 0
    point_batch_size = int(args.point_batch_size or training_cfg.get("batch_size", 32768))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            pred_scaled_parts = []
            for start in range(0, batch.num_points, point_batch_size):
                end = min(start + point_batch_size, batch.num_points)
                pred_scaled_parts.append(model(batch.features[start:end]))
            pred_scaled = torch.cat(pred_scaled_parts, dim=0)
            _pred_log, pred_raw = _decode_prediction(pred_scaled, y_scaler)

            pred_np = pred_raw.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
            target_np = batch.target_raw.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
            node_np = batch.node_indices.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            sample_np = batch.sample_index.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            frequencies = batch.frequency_hz.detach().cpu().numpy().astype(np.float64, copy=False)

            for local_idx, sample_name in enumerate(batch.names):
                case_name = batch.case_names[local_idx]
                mask = sample_np == local_idx
                if not np.any(mask):
                    continue
                accum = case_accum.setdefault(
                    case_name,
                    {"frequencies": [], "nodes": [], "target": [], "pred": []},
                )
                accum["frequencies"].append(float(frequencies[local_idx]))
                accum["nodes"].append(node_np[mask].copy())
                accum["target"].append(np.clip(target_np[mask], 0.0, None).copy())
                accum["pred"].append(np.clip(pred_np[mask], 0.0, None).copy())
                sample_count += 1

            if batch_idx == 1 or batch_idx % int(training_cfg.get("eval_progress_every_steps", 30)) == 0:
                logger.info("Progress | loader_batch=%s | samples=%s | cases_seen=%s", batch_idx, sample_count, len(case_accum))

    all_case_summaries: list[dict[str, Any]] = []
    all_band_summaries: list[dict[str, Any]] = []
    combined_top_rows: list[pd.DataFrame] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        accum = case_accum.get(case_name)
        if not accum:
            logger.warning("No accumulated samples for case: %s", case_name)
            continue
        frequencies = np.array(accum["frequencies"], dtype=np.float64)
        weights = _trapz_weights(frequencies)
        node_table = _read_node_table(case_dir)
        selected_nodes = np.unique(np.concatenate(accum["nodes"]).astype(np.int64, copy=False))
        node_position = {int(node): idx for idx, node in enumerate(selected_nodes.tolist())}
        target_integral = np.zeros(len(selected_nodes), dtype=np.float64)
        pred_integral = np.zeros(len(selected_nodes), dtype=np.float64)

        for frame_idx, weight in enumerate(weights):
            if weight <= 0.0:
                continue
            nodes = accum["nodes"][frame_idx]
            positions = np.fromiter((node_position[int(node)] for node in nodes), dtype=np.int64, count=len(nodes))
            target_integral[positions] += accum["target"][frame_idx] * weight
            pred_integral[positions] += accum["pred"][frame_idx] * weight

        rows = pd.DataFrame(
            {
                "case_name": case_name,
                "node_index": selected_nodes,
                "target_rmises": np.sqrt(np.clip(target_integral, 0.0, None)),
                "pred_rmises": np.sqrt(np.clip(pred_integral, 0.0, None)),
            }
        )
        rows = rows.merge(node_table, on="node_index", how="left")
        rows = _merge_native_rmises(case_dir, rows)
        rows["abs_error"] = (rows["pred_rmises"] - rows["target_rmises"]).abs()
        rows["relative_error"] = _safe_relative_error(rows["pred_rmises"], rows["target_rmises"])
        rows["pred_target_ratio"] = rows["pred_rmises"] / rows["target_rmises"].abs().where(rows["target_rmises"].abs() > 1e-12, np.nan)
        rows = rows.sort_values("target_rmises", ascending=False, kind="mergesort").reset_index(drop=True)
        rows["target_rank"] = np.arange(1, len(rows) + 1, dtype=np.int64)
        rows["target_rank_fraction"] = rows["target_rank"] / max(len(rows), 1)

        top_rows = []
        for group_name, top_n in SUMMARY_NODE_COUNTS.items():
            group = rows.head(min(top_n, len(rows))).copy()
            group.insert(1, "top_group", group_name)
            top_rows.append(group)
            node_summary = _summarize_group(group, group_name)
            node_summary.update({"case_name": case_name, "top_node_count": int(top_n)})
            all_band_summaries.append(node_summary)
        for band_name, fraction in SUMMARY_BANDS.items():
            count = _top_count(len(rows), fraction)
            group = rows.head(count)
            band_summary = _summarize_group(group, band_name)
            band_summary.update({"case_name": case_name, "top_fraction": fraction})
            all_band_summaries.append(band_summary)

        case_summary = _summarize_group(rows, "all_selected_nodes")
        case_summary.update(
            {
                "case_name": case_name,
                "frequency_count": int(len(frequencies)),
                "frequency_min_hz": float(np.min(frequencies)),
                "frequency_max_hz": float(np.max(frequencies)),
            }
        )
        all_case_summaries.append(case_summary)

        top_case_rows = pd.concat(top_rows, ignore_index=True)
        top_case_rows.to_csv(cases_dir / f"{case_name}_top_nodes.csv", index=False)
        combined_top_rows.append(top_case_rows)
        if args.write_per_node:
            rows.to_csv(cases_dir / f"{case_name}_final_rmises_per_node.csv", index=False)

    case_summary_df = pd.DataFrame(all_case_summaries)
    band_summary_df = pd.DataFrame(all_band_summaries)
    case_summary_df.to_csv(output_dir / "case_summary.csv", index=False)
    band_summary_df.to_csv(output_dir / "top_band_summary_by_case.csv", index=False)
    if combined_top_rows:
        pd.concat(combined_top_rows, ignore_index=True).to_csv(output_dir / "top_nodes_by_case.csv", index=False)

    aggregate: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "cases": int(len(all_case_summaries)),
        "samples": int(sample_count),
        "integration_rule": "trapezoidal over evaluated positive-frequency MISES_psd_density frames",
        "node_scope": dataset_cfg.get("node_scope") or ("earpiece" if dataset_cfg.get("earpiece_region", {}).get("enabled", True) else "all_nodes"),
        "case_mean": {},
        "top_band_case_mean": {},
    }
    if not case_summary_df.empty:
        numeric_cols = case_summary_df.select_dtypes(include=[np.number]).columns
        aggregate["case_mean"] = {col: float(case_summary_df[col].mean()) for col in numeric_cols}
    if not band_summary_df.empty:
        for group, group_df in band_summary_df.groupby("group", sort=True):
            numeric_cols = group_df.select_dtypes(include=[np.number]).columns
            aggregate["top_band_case_mean"][str(group)] = {col: float(group_df[col].mean()) for col in numeric_cols}
    write_json(output_dir / "summary.json", aggregate)
    logger.info("Saved final RMises top-k summary: %s", output_dir)


if __name__ == "__main__":
    main()

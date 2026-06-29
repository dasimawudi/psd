"""Compute top-percentile evaluation metrics for the model report.

Top-percent groups: top1%, top2%, top5%, top10%, top15%, top25%
Metrics per group: within25, relative MAE, relative log MAE
Regions: overall (all points), earpiece_region, disk_region

Usage:
  python node/eval_top_percent_metrics.py \
    --checkpoint <path/to/best.pt> \
    --output-dir <path/to/output>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from case7_node_mlp.data import (
    discover_case_index,
    resolve_case_splits,
    expand_case_sample_paths,
)
from case7_node_mlp.runtime import read_config, resolve_device, ensure_dir, make_logger
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import (
    make_loader,
    build_model,
    regression_output,
    _point_chunks,
)

TOP_GROUPS = [
    ("top1%", 0.0, 0.01),
    ("top2%", 0.0, 0.02),
    ("top5%", 0.0, 0.05),
    ("top10%", 0.0, 0.10),
    ("top15%", 0.0, 0.15),
    ("top25%", 0.0, 0.25),
]


def _empty_counts():
    return {"points": 0.0, "rel_points": 0.0, "within25": 0.0, "rel_sum": 0.0, "log_sum": 0.0}


def _add_counts(cnts, pred, target, pred_log, target_log):
    """Add per-point metrics. pred/target are 1D tensors for selected points."""
    n = float(pred.numel())
    cnts["points"] += n
    abs_err = (pred - target).abs()
    log_abs_err = (pred_log - target_log).abs()
    rel_mask = target.abs() > 1e-12
    cnts["rel_points"] += float(rel_mask.sum().item())
    if bool(rel_mask.any()):
        rel_err = abs_err[rel_mask] / target[rel_mask].abs()
        cnts["within25"] += float((rel_err <= 0.25).sum().item())
        cnts["rel_sum"] += float(rel_err.sum().item())
    cnts["log_sum"] += float(log_abs_err.sum().item())


def _finalize(cnts):
    pts = max(cnts["points"], 1.0)
    rel_pts = max(cnts["rel_points"], 1.0)
    return dict(
        points=int(pts),
        within25=cnts["within25"] / rel_pts,
        relative_mae=cnts["rel_sum"] / rel_pts,
        relative_log_mae=cnts["log_sum"] / pts,
    )


def _rank_fraction(values: torch.Tensor) -> torch.Tensor:
    """Per-sample rank fraction: 1/n (highest target) to 1.0 (lowest target)."""
    values = values.reshape(-1)
    n = int(values.numel())
    rf = torch.empty(n, dtype=torch.float32, device=values.device)
    order = torch.argsort(values, descending=True, stable=True)
    rf[order] = torch.arange(1, n + 1, dtype=torch.float32, device=values.device) / float(n)
    return rf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4000000)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else (ckpt_path.parent / "resolved_config.yaml")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Config: {cfg_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = read_config(str(cfg_path))
    device = resolve_device(str(args.device))

    dataset_cfg = dict(cfg.get("dataset", {}))
    feature_cfg = dict(cfg.get("features", {}))
    target_cfg = dict(cfg.get("target", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    root = str(dataset_cfg.get("root", ""))

    x_scaler = StandardScaler.from_state_dict(ckpt["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(ckpt["y_scaler"])
    feature_schema = dict(ckpt.get("feature_schema", {}))
    input_dim = int(feature_schema.get("input_dim", 0))

    model = build_model(cfg, input_dim=input_dim, feature_schema=feature_schema)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("metrics", {}).get("epoch", None)

    case_index = discover_case_index(root)
    split_map = resolve_case_splits(root, dataset_cfg=dataset_cfg)
    eval_dirs = [case_index[name] for name in split_map.get(args.split, [])]
    eval_paths = expand_case_sample_paths(eval_dirs, dataset_cfg)
    print(f"Split: {args.split} | Samples: {len(eval_paths)}")

    loader = make_loader(
        sample_paths=eval_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=args.sample_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
        pin_memory=True,
    )

    output_dir = ensure_dir(args.output_dir)

    # Accumulators:
    #   overall_cnts[grp_name] -> counts for all points in that top group
    #   region_cnts[region][grp_name] -> counts for region points in that top group
    #   overall_all -> counts for all points (no top filter)
    #   region_all[region] -> counts for all points in region (no top filter)
    overall_cnts = {g: _empty_counts() for g, _, _ in TOP_GROUPS}
    overall_all = _empty_counts()
    region_cnts = {
        r: {g: _empty_counts() for g, _, _ in TOP_GROUPS}
        for r in ["earpiece_region", "disk_region"]
    }
    region_all = {r: _empty_counts() for r in ["earpiece_region", "disk_region"]}

    total_batches = len(loader)
    started_at = time.monotonic()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if batch.num_points <= 0:
                continue

            # Inference
            pred_scaled_parts = []
            for chunk in _point_chunks(batch, point_batch_size=args.batch_size, shuffle=False):
                feats = batch.features[chunk].to(device, non_blocking=True)
                pred_scaled_parts.append(regression_output(model(feats)).detach())
            pred_scaled = torch.cat(pred_scaled_parts, dim=0)

            y_mean = y_scaler.mean.to(device)
            y_std = y_scaler.std.to(device)
            pred_log_all = (pred_scaled * y_std + y_mean).reshape(-1)
            pred_raw_all = torch.expm1(pred_log_all.clamp_max(20.0))
            target_raw_all = batch.target_raw.to(device, non_blocking=True).reshape(-1)
            target_log_all = target_raw_all.clamp_min(0.0).add(1.0).log()
            sample_idx_all = batch.sample_index.to(device, non_blocking=True).reshape(-1)

            # Region masks
            region_mask_dict = {}
            if batch.region_masks:
                for rname in ["earpiece_region", "disk_region"]:
                    if rname in batch.region_masks:
                        region_mask_dict[rname] = (
                            batch.region_masks[rname]
                            .to(device, non_blocking=True)
                            .to(dtype=torch.bool)
                            .reshape(-1)
                        )

            for s_idx in range(len(batch.names)):
                s_mask = sample_idx_all == s_idx
                if not bool(s_mask.any()):
                    continue

                s_target = target_raw_all[s_mask]
                s_pred = pred_raw_all[s_mask]
                s_pred_log = pred_log_all[s_mask]
                s_target_log = target_log_all[s_mask]

                # Overall (all points, no filter)
                _add_counts(overall_all, s_pred, s_target, s_pred_log, s_target_log)

                # Overall top groups
                s_rf = _rank_fraction(s_target)
                for grp_name, left, right in TOP_GROUPS:
                    top_mask = (s_rf > left) & (s_rf <= right)
                    if bool(top_mask.any()):
                        _add_counts(
                            overall_cnts[grp_name],
                            s_pred[top_mask],
                            s_target[top_mask],
                            s_pred_log[top_mask],
                            s_target_log[top_mask],
                        )

                # Region
                for rname in ["earpiece_region", "disk_region"]:
                    r_mask_tensor = region_mask_dict.get(rname)
                    if r_mask_tensor is None:
                        continue
                    sr_mask = r_mask_tensor[s_mask]
                    if not bool(sr_mask.any()):
                        continue
                    sr_target = s_target[sr_mask]
                    sr_pred = s_pred[sr_mask]
                    sr_pred_log = s_pred_log[sr_mask]
                    sr_target_log = s_target_log[sr_mask]

                    # Region all points
                    _add_counts(region_all[rname], sr_pred, sr_target, sr_pred_log, sr_target_log)

                    # Region top groups
                    sr_rf = _rank_fraction(sr_target)
                    for grp_name, left, right in TOP_GROUPS:
                        top_mask = (sr_rf > left) & (sr_rf <= right)
                        if bool(top_mask.any()):
                            _add_counts(
                                region_cnts[rname][grp_name],
                                sr_pred[top_mask],
                                sr_target[top_mask],
                                sr_pred_log[top_mask],
                                sr_target_log[top_mask],
                            )

            if step % args.progress_every == 0 or step == total_batches:
                elapsed = time.monotonic() - started_at
                print(f"  {step}/{total_batches} batches | {elapsed:.0f}s", flush=True)

    # ── Finalize ──────────────────────────────────────────────────────
    overall_metrics = {g: _finalize(overall_cnts[g]) for g, _, _ in TOP_GROUPS}
    overall_all_metrics = _finalize(overall_all)
    region_metrics = {
        r: {g: _finalize(region_cnts[r][g]) for g, _, _ in TOP_GROUPS}
        for r in ["earpiece_region", "disk_region"]
    }
    region_all_metrics = {r: _finalize(region_all[r]) for r in ["earpiece_region", "disk_region"]}

    # Print debug
    print(f"\nOverall all points: pts={overall_all['points']:.0f} rel_pts={overall_all['rel_points']:.0f} "
          f"within25_sum={overall_all['within25']:.1f} rel_sum={overall_all['rel_sum']:.1f}")

    # ── Print results ─────────────────────────────────────────────────
    header = f"{'Region':<22} {'Top':>8} {'Points':>12} {'within25':>10} {'相对MAE':>10} {'相对log MAE':>13}"
    sep = "-" * 75

    print("\n" + sep)
    print("MODEL EVALUATION REPORT")
    print(sep)

    # Overall (all points, no filter) row
    m = overall_all_metrics
    print(f"{'overall (全零件)':<22} {'overall':>8} {m['points']:>12,d} "
          f"{m['within25']:>9.2%} {m['relative_mae']:>9.2%} {m['relative_log_mae']:>12.4f}")

    # Top percent rows for overall
    print(sep)
    print(header)
    print(sep)
    for grp_name, _, _ in TOP_GROUPS:
        m = overall_metrics[grp_name]
        print(f"{'overall':<22} {grp_name:>8} {m['points']:>12,d} "
              f"{m['within25']:>9.2%} {m['relative_mae']:>9.2%} {m['relative_log_mae']:>12.4%}")

    # Regions
    for rname in ["earpiece_region", "disk_region"]:
        r_display = "耳片区域" if rname == "earpiece_region" else "圆盘区域"
        print(sep)
        m = region_all_metrics[rname]
        print(f"{r_display:<22} {'overall':>8} {m['points']:>12,d} "
              f"{m['within25']:>9.2%} {m['relative_mae']:>9.2%} {m['relative_log_mae']:>12.4f}")
        print(sep)
        print(header)
        print(sep)
        for grp_name, _, _ in TOP_GROUPS:
            m = region_metrics[rname][grp_name]
            print(f"{r_display:<22} {grp_name:>8} {m['points']:>12,d} "
                  f"{m['within25']:>9.2%} {m['relative_mae']:>9.2%} {m['relative_log_mae']:>12.4%}")

    # ── Save JSON & CSV ───────────────────────────────────────────────
    json_data = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "split": args.split,
        "eval_samples": len(eval_paths),
        "overall": overall_all_metrics,
        "overall_top": overall_metrics,
        "earpiece_region": region_all_metrics["earpiece_region"],
        "earpiece_region_top": region_metrics["earpiece_region"],
        "disk_region": region_all_metrics["disk_region"],
        "disk_region_top": region_metrics["disk_region"],
    }
    json_path = output_dir / f"{args.split}_top_percent_relative_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON: {json_path}")

    csv_path = output_dir / f"{args.split}_top_percent_relative_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "top_group", "points", "within25", "relative_mae", "relative_log_mae"])
        # Overall (all points)
        m = overall_all_metrics
        w.writerow(["overall", "overall", m["points"], round(m["within25"], 6),
                     round(m["relative_mae"], 6), round(m["relative_log_mae"], 6)])
        # Overall top groups
        for grp_name, _, _ in TOP_GROUPS:
            m = overall_metrics[grp_name]
            w.writerow(["overall", grp_name, m["points"], round(m["within25"], 6),
                         round(m["relative_mae"], 6), round(m["relative_log_mae"], 6)])
        # Regions
        for rname in ["earpiece_region", "disk_region"]:
            r_display = "耳片区域" if rname == "earpiece_region" else "圆盘区域"
            m = region_all_metrics[rname]
            w.writerow([r_display, "overall", m["points"], round(m["within25"], 6),
                         round(m["relative_mae"], 6), round(m["relative_log_mae"], 6)])
            for grp_name, _, _ in TOP_GROUPS:
                m = region_metrics[rname][grp_name]
                w.writerow([r_display, grp_name, m["points"], round(m["within25"], 6),
                             round(m["relative_mae"], 6), round(m["relative_log_mae"], 6)])
    print(f"CSV: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()

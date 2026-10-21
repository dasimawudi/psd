"""Quick script to evaluate continue32 best checkpoint with full per-region top-k."""
import sys
sys.path.insert(0, "node")

from pathlib import Path
from case7_node_mlp.runtime import read_config, resolve_device, make_logger, ensure_dir
from case7_node_mlp.trainer import build_model, evaluate, make_loader
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.data import discover_case_index, resolve_case_splits, expand_case_sample_paths
import torch
import json

CHECKPOINT = "node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32/best.pt"
OUTPUT_DIR = "node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32/eval_val_full_topk"
SPLIT = "val"

checkpoint = torch.load(CHECKPOINT, map_location="cpu")
config = dict(checkpoint["config"])
dataset_cfg = dict(config["dataset"])
feature_cfg = dict(config.get("features", {}))
training_cfg = dict(config.get("training", {}))

output_dir = ensure_dir(OUTPUT_DIR)
logger = make_logger(output_dir, logger_name="quick_eval_full_topk", log_file="evaluate.log")
device = resolve_device("auto")

feature_schema = dict(checkpoint["feature_schema"])
x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
model = build_model(config, input_dim=int(feature_schema["input_dim"]), feature_schema=feature_schema).to(device)
model.load_state_dict(checkpoint["model_state"])

case_index = discover_case_index(dataset_cfg["root"])
split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
case_dirs = [case_index[name] for name in split_names[SPLIT]]
sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
logger.info("Evaluating with full topk | split=%s | cases=%s | samples=%s", SPLIT, len(case_dirs), len(sample_paths))

target_cfg = dict(config.get("target", {}))
loss_cfg = dict(config.get("loss", {}))

loader = make_loader(
    sample_paths=sample_paths,
    dataset_cfg=dataset_cfg,
    feature_cfg=feature_cfg,
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    feature_schema=feature_schema,
    target_cfg=target_cfg,
    loss_cfg=loss_cfg,
    sample_batch_size=16,  # small batch to keep memory low
    num_workers=2,
    shuffle=False,
    persistent_workers=False,
    prefetch_factor=1,
)

point_batch_size = 8192  # smaller to avoid OOM

result = evaluate(
    model=model,
    loader=loader,
    y_scaler=y_scaler,
    device=device,
    point_batch_size=point_batch_size,
    split_name=SPLIT,
    epoch=32,
    logger=logger,
    topk_mode="full",
    include_region_metrics=True,
)

# Save per-region topk
print("\n=== Per-Region Top-K ===")
regions = ["fullpart_stress", "earpiece_region_stress", "disk_stress", "disk_center_stress"]
for r in regions:
    print(f"\n--- {r} ---")
    for suffix in ["", "_top1", "_top1_5", "_top5", "_top5_10", "_top10"]:
        k_log = f"{r}{suffix}_log_mae"
        k_w25 = f"{r}{suffix}_within25_ratio"
        k_pt = f"{r}{suffix}_pred_target_ratio"
        vals = []
        if k_log in result.metrics:
            vals.append(f"log_mae={result.metrics[k_log]:.4f}")
        if k_w25 in result.metrics:
            vals.append(f"within25={result.metrics[k_w25]:.4f}")
        if k_pt in result.metrics:
            vals.append(f"pred/target={result.metrics[k_pt]:.4f}")
        if vals:
            label = suffix if suffix else "overall"
            print(f"  {label}: {' | '.join(vals)}")

# Save metrics
metrics_path = output_dir / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(result.metrics, f, indent=2)
print(f"\nSaved metrics to {metrics_path}")

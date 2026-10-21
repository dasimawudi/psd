#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-yolo}"
DEVICE="${DEVICE:-cuda:6}"
SPLIT="${SPLIT:-all}"

CHECKPOINT="node/outputs/node_mlp_v5_exp_region_distance_within25_50ep/best.pt"
OUTPUT_DIR="node/outputs/node_mlp_v5_exp_region_distance_within25_50ep/target_quantile_within25_p1_all"

PYTHONPATH=node conda run -n "$CONDA_ENV" python node/plot_disk_center_p1_within25.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --split "$SPLIT" \
  --region-label Earpiece \
  "$@"

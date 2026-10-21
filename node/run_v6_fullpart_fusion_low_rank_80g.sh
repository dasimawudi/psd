#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data1/libo/miniconda3/envs/ci2n/bin/python}"
CONFIG="${CONFIG:-node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g.yaml}"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH=node \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" node/train_node_mlp.py \
  --config "$CONFIG"

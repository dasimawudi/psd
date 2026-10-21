#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
CONFIG="${CONFIG:-node/configs/node_mlp_v6_fullpart_fusion_ear_region_loss_80g.yaml}"
PYTHON="${PYTHON:-/data1/libo/miniconda3/envs/ci2n/bin/python}"
OUT_DIR="${OUT_DIR:-node/outputs/node_mlp_v6_fullpart_fusion_ear_region_loss_80g}"

mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PYTHONPATH:-node}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$PYTHON" node/train_node_mlp.py --config "$CONFIG"

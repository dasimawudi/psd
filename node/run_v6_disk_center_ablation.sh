#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_NAME=${1:-node_mlp_v6_disk_center_baseline.yaml}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
PYTHONPATH=node \
conda run -n ci2n python node/train_node_mlp.py \
  --config "node/configs/${CONFIG_NAME}"

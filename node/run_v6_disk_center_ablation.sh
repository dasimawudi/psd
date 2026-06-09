#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_NAME=${1:-node_mlp_v6_disk_center_baseline.yaml}
CONDA_ENV=${CONDA_ENV:-yolo}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
PYTHONPATH=node \
conda run --no-capture-output -n "${CONDA_ENV}" python node/train_node_mlp.py \
  --config "node/configs/${CONFIG_NAME}"

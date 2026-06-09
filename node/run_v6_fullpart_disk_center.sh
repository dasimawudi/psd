#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
PYTHONPATH=node \
conda run -n ci2n python node/train_node_mlp.py \
  --config node/configs/node_mlp_v6_fullpart_disk_center.yaml

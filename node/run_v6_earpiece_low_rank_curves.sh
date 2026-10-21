#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
CONDA_ENV="${CONDA_ENV:-yolo}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_rank5() {
  CUDA_VISIBLE_DEVICES="${EARPIECE_LOW_RANK_GPU:-0}" \
  PYTHONPATH=node \
  conda run -n "$CONDA_ENV" "$PYTHON_BIN" node/train_node_mlp.py \
    --config node/configs/node_mlp_v6_earpiece_low_rank_curves_fast_eval.yaml
}

run_rank8() {
  CUDA_VISIBLE_DEVICES="${EARPIECE_LOW_RANK_RANK8_GPU:-5}" \
  PYTHONPATH=node \
  conda run -n "$CONDA_ENV" "$PYTHON_BIN" node/train_node_mlp.py \
    --config node/configs/node_mlp_v6_earpiece_low_rank_curves_rank8_fast_eval.yaml
}

case "$STAGE" in
  rank5)
    run_rank5
    ;;
  rank8)
    run_rank8
    ;;
  all)
    run_rank5
    run_rank8
    ;;
  *)
    echo "Usage: $0 [rank5|rank8|all]" >&2
    exit 2
    ;;
esac

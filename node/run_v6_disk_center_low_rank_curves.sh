#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
CONDA_ENV="${CONDA_ENV:-ci2n}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_diag() {
  PYTHONPATH=node conda run -n "$CONDA_ENV" "$PYTHON_BIN" node/center_curve_low_rank_diagnostics.py \
    --config node/configs/node_mlp_v6_disk_center_hotspot_features.yaml \
    --split test \
    --num-cases "${LOW_RANK_DIAG_CASES:-32}" \
    --output-dir node/outputs/node_mlp_v6_disk_center_low_rank_diagnostics
}

run_rank() {
  CUDA_VISIBLE_DEVICES="${LOW_RANK_GPU:-4}" \
  PYTHONPATH=node \
  conda run -n "$CONDA_ENV" "$PYTHON_BIN" node/train_node_mlp.py \
    --config node/configs/node_mlp_v6_disk_center_low_rank_curves.yaml
}

run_calibrated() {
  CUDA_VISIBLE_DEVICES="${CALIBRATED_GPU:-7}" \
  PYTHONPATH=node \
  conda run -n "$CONDA_ENV" "$PYTHON_BIN" node/train_node_mlp.py \
    --config node/configs/node_mlp_v6_disk_center_low_rank_curves_calibrated.yaml
}

case "$STAGE" in
  diag)
    run_diag
    ;;
  rank)
    run_rank
    ;;
  calibrated)
    run_calibrated
    ;;
  all)
    run_diag
    run_rank
    run_calibrated
    ;;
  *)
    echo "Usage: $0 [diag|rank|calibrated|all]" >&2
    exit 2
    ;;
esac

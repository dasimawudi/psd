#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT=.cache/mises_psd_next1000_full_export_step0p5_balanced_v1
CHECKPOINT=node/outputs/node_mlp_v6_fullpart_disk_center/best.pt

test -d "${DATA_ROOT}" || {
  echo "Dataset not found: ${DATA_ROOT}" >&2
  exit 1
}
test -f "${CHECKPOINT}" || {
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
}

exec bash node/run_v6_fullpart_disk_center.sh \
  node_mlp_v6_fullpart_disk_center_opt.yaml

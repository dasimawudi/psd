#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} PYTHONPATH=node conda run -n ci2n python -m train7.train --config node/train7/configs/v8_within25_zero_peak.yaml

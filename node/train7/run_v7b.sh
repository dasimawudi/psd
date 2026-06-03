#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=node conda run -n ci2n python -m train7.train --config node/train7/configs/v7b_zero_gate.yaml

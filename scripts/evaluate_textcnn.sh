#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluate --config configs/textcnn.yaml --split test --device "${DEVICE:-cuda}"

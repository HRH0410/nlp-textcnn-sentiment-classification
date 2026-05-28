#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/fasttext.yaml --device "${DEVICE:-cuda}"

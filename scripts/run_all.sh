#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/fasttext.yaml --device "${DEVICE:-cuda}"
python -m src.evaluate --config configs/fasttext.yaml --split test --device "${DEVICE:-cuda}"
python -m src.train --config configs/textcnn.yaml --device "${DEVICE:-cuda}"
python -m src.evaluate --config configs/textcnn.yaml --split test --device "${DEVICE:-cuda}"

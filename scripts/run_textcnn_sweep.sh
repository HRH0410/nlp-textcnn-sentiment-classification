#!/usr/bin/env bash
set -euo pipefail

python -m src.download_glove --output-dir embeddings --dim 300

for config in \
  configs/textcnn.yaml \
  configs/textcnn_glove_small.yaml \
  configs/textcnn_glove_weighted.yaml
do
  python -m src.train --config "$config" --device "${DEVICE:-cuda}"
  python -m src.evaluate --config "$config" --split test --device "${DEVICE:-cuda}"
done

python -m src.summarize_results

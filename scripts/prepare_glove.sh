#!/usr/bin/env bash
set -euo pipefail

python -m src.download_glove --output-dir embeddings --dim 300

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pip install -U pip
pip install -r requirements.txt

python scripts/train_colab.py \
  --epochs "${EPOCHS:-6}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --synthetic-samples "${SYNTHETIC_SAMPLES:-40000}" \
  --output-dir "${OUTPUT_DIR:-artifacts/stride_moe}"

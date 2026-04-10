#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/stride_moe}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/content/ittamt_datasets}"
HF_HOME="${HF_HOME:-$DATASET_CACHE_DIR/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATASET_CACHE_DIR/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DATASET_CACHE_DIR/hub}"

export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

python -m pip install -U pip
pip install -r requirements.txt

python scripts/train_colab.py \
  --epochs "${EPOCHS:-6}" \
  --batch-size "${BATCH_SIZE:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --prefetch-factor "${PREFETCH_FACTOR:-2}" \
  --synthetic-samples "${SYNTHETIC_SAMPLES:-40000}" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --output-dir "$OUTPUT_DIR"

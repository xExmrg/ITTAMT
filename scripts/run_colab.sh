#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mount_google_drive() {
  python - <<'PY'
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
except Exception as exc:  # pragma: no cover - only used in Colab
    print(f"google drive mount unavailable: {exc}")
else:
    mount_root = Path("/content/drive")
    my_drive = mount_root / "MyDrive"
    if my_drive.exists():
        print(f"google drive already mounted at {mount_root}")
    else:
        drive.mount(str(mount_root), force_remount=False)
        print(f"google drive mounted at {mount_root}")
PY
}

if [[ "${MOUNT_GOOGLE_DRIVE:-1}" == "1" && -d /content ]]; then
  mount_google_drive
fi

if [[ -d /content/drive/MyDrive ]]; then
  DEFAULT_PERSIST_ROOT="/content/drive/MyDrive/ittamt"
else
  DEFAULT_PERSIST_ROOT="/content/ittamt_persist"
fi

PERSIST_ROOT="${PERSIST_ROOT:-$DEFAULT_PERSIST_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$PERSIST_ROOT/artifacts/stride_moe}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$PERSIST_ROOT/datasets}"
HF_HOME="${HF_HOME:-$DATASET_CACHE_DIR/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATASET_CACHE_DIR/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DATASET_CACHE_DIR/hub}"
TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/torch}"

export HF_HOME
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export TORCH_HOME

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

echo "persistent storage root: $PERSIST_ROOT"
echo "dataset cache dir: $HF_DATASETS_CACHE"
echo "output dir: $OUTPUT_DIR"

python -m pip install -U pip
pip install -r requirements.txt

python scripts/train_colab.py \
  --epochs "${EPOCHS:-6}" \
  --batch-size "${BATCH_SIZE:-0}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --synthetic-samples "${SYNTHETIC_SAMPLES:-40000}" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --output-dir "$OUTPUT_DIR"

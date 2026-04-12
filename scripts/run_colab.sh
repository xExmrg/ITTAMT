#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

sync_dir() {
  local src="$1"
  local dst="$2"
  local mode="${3:-update}"

  if [[ ! -d "$src" ]]; then
    return 0
  fi
  mkdir -p "$dst"

  if command -v rsync >/dev/null 2>&1; then
    if [[ "$mode" == "hydrate" ]]; then
      rsync -a --ignore-existing "$src"/ "$dst"/
    else
      rsync -a "$src"/ "$dst"/
    fi
    return 0
  fi

  python - "$src" "$dst" "$mode" <<'PY'
from __future__ import annotations

import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
mode = sys.argv[3]

if not src.exists():
    raise SystemExit(0)

for path in src.rglob("*"):
    rel = path.relative_to(src)
    target = dst / rel
    if path.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hydrate" and target.exists():
        continue
    shutil.copy2(path, target)
PY
}

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

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi

  local token=""
  if [[ -d /content ]]; then
    token="$(python - <<'PY'
try:
    from google.colab import userdata  # type: ignore
except Exception:
    raise SystemExit(0)

try:
    token = userdata.get("HF_TOKEN")
except Exception:
    token = ""

if token:
    print(token)
PY
)"
  fi

  if [[ -z "$token" && -f "$PERSIST_ROOT/.hf_token" ]]; then
    token="$(<"$PERSIST_ROOT/.hf_token")"
  fi

  if [[ -n "$token" ]]; then
    export HF_TOKEN="$token"
  fi
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
PERSIST_OUTPUT_DIR="${PERSIST_OUTPUT_DIR:-$PERSIST_ROOT/artifacts/stride_moe}"
PERSIST_DATASET_CACHE_DIR="${PERSIST_DATASET_CACHE_DIR:-$PERSIST_ROOT/datasets}"
PERSIST_HF_HOME="${PERSIST_HF_HOME:-$PERSIST_DATASET_CACHE_DIR/hf_home}"
PERSIST_HF_DATASETS_CACHE="${PERSIST_HF_DATASETS_CACHE:-$PERSIST_DATASET_CACHE_DIR/datasets}"
PERSIST_HUGGINGFACE_HUB_CACHE="${PERSIST_HUGGINGFACE_HUB_CACHE:-$PERSIST_DATASET_CACHE_DIR/hub}"
PERSIST_TORCH_HOME="${PERSIST_TORCH_HOME:-$PERSIST_ROOT/torch}"

if [[ -d /content ]]; then
  DEFAULT_LOCAL_RUNTIME_ROOT="/content/ittamt_runtime"
else
  DEFAULT_LOCAL_RUNTIME_ROOT="$ROOT_DIR/.ittamt_runtime"
fi

LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-$DEFAULT_LOCAL_RUNTIME_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_RUNTIME_ROOT/artifacts/stride_moe}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$LOCAL_RUNTIME_ROOT/datasets}"
HF_HOME="${HF_HOME:-$DATASET_CACHE_DIR/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATASET_CACHE_DIR/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DATASET_CACHE_DIR/hub}"
TORCH_HOME="${TORCH_HOME:-$LOCAL_RUNTIME_ROOT/torch}"

mkdir -p \
  "$OUTPUT_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TORCH_HOME" \
  "$PERSIST_OUTPUT_DIR" \
  "$PERSIST_HF_HOME" \
  "$PERSIST_HF_DATASETS_CACHE" \
  "$PERSIST_HUGGINGFACE_HUB_CACHE" \
  "$PERSIST_TORCH_HOME"

echo "persistent storage root: $PERSIST_ROOT"
echo "local runtime root: $LOCAL_RUNTIME_ROOT"
echo "runtime dataset cache dir: $HF_DATASETS_CACHE"
echo "runtime output dir: $OUTPUT_DIR"
echo "persistent output dir: $PERSIST_OUTPUT_DIR"

echo "hydrating local runtime caches from persistent storage..."
sync_dir "$PERSIST_HF_HOME" "$HF_HOME" hydrate
sync_dir "$PERSIST_HF_DATASETS_CACHE" "$HF_DATASETS_CACHE" hydrate
sync_dir "$PERSIST_HUGGINGFACE_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" hydrate
sync_dir "$PERSIST_TORCH_HOME" "$TORCH_HOME" hydrate
sync_dir "$PERSIST_OUTPUT_DIR" "$OUTPUT_DIR" hydrate

persist_back() {
  echo "syncing runtime caches and artifacts back to persistent storage..."
  sync_dir "$HF_HOME" "$PERSIST_HF_HOME" update
  sync_dir "$HF_DATASETS_CACHE" "$PERSIST_HF_DATASETS_CACHE" update
  sync_dir "$HUGGINGFACE_HUB_CACHE" "$PERSIST_HUGGINGFACE_HUB_CACHE" update
  sync_dir "$TORCH_HOME" "$PERSIST_TORCH_HOME" update
  sync_dir "$OUTPUT_DIR" "$PERSIST_OUTPUT_DIR" update
}

trap persist_back EXIT

load_hf_token

export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE
export HUGGINGFACE_HUB_CACHE
export TORCH_HOME
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

python -m pip install -U pip
pip install -r requirements.txt

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "hugging face auth: enabled"
else
  echo "hugging face auth: missing HF_TOKEN, first-time downloads may be rate-limited"
fi
echo "hf_xet_high_performance: ${HF_XET_HIGH_PERFORMANCE}"

python scripts/train_colab.py \
  --epochs "${EPOCHS:-6}" \
  --batch-size "${BATCH_SIZE:-0}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --synthetic-samples "${SYNTHETIC_SAMPLES:-40000}" \
  --use-iam "${USE_IAM:-0}" \
  --use-iiit5k "${USE_IIIT5K:-1}" \
  --use-textocr "${USE_TEXTOCR:-0}" \
  --use-sroie "${USE_SROIE:-0}" \
  --use-cord "${USE_CORD:-0}" \
  --use-funsd "${USE_FUNSD:-0}" \
  --use-doclaynet "${USE_DOCLAYNET:-0}" \
  --use-xfund "${USE_XFUND:-0}" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --mirror-output-dir "$PERSIST_OUTPUT_DIR"

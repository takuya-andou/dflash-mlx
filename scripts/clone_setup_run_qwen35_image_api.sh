#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/takuya-andou/dflash-mlx.git}"
BRANCH="${BRANCH:-feature/qwen35-image-support}"
WORKDIR="${WORKDIR:-$HOME/work}"
REPO_DIR="${REPO_DIR:-$WORKDIR/dflash-mlx}"

TARGET_MODEL="${TARGET_MODEL:-mlx-community/Qwen3.5-4B-MLX-bf16}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SEED="${SEED:-0}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script requires Apple Silicon macOS because dflash-mlx depends on MLX." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but was not found." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but was not found." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$WORKDIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin "$BRANCH"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "origin/$BRANCH"
fi

git pull --ff-only origin "$BRANCH"

uv sync

echo "[download] target=$TARGET_MODEL"
echo "[download] draft=$DRAFT_MODEL"
TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" SEED="$SEED" \
uv run --with pillow --with transformers python - <<'PY'
import os

from dflash_mlx import DFlashGenerator

runner = DFlashGenerator(
    target_model=os.environ["TARGET_MODEL"],
    draft_model=os.environ["DRAFT_MODEL"],
    seed=int(os.environ["SEED"]),
)
print(f"[ready] target={runner.target_model_path}")
print(f"[ready] draft={runner.draft_path}")
PY

cat <<EOF
[start] launching API server on http://$HOST:$PORT
[start] example:
curl -X POST "http://$HOST:$PORT/generate" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Describe the image.",
    "images": ["https://example.com/example.jpg"],
    "max_new_tokens": 128
  }'

[start] upload example:
curl -X POST "http://$HOST:$PORT/generate-upload" \\
  -F 'prompt=Describe the image.' \\
  -F 'images=@/absolute/path/to/example.jpg' \\
  -F 'max_new_tokens=128'
EOF

TARGET_MODEL="$TARGET_MODEL" \
DRAFT_MODEL="$DRAFT_MODEL" \
HOST="$HOST" \
PORT="$PORT" \
SEED="$SEED" \
uv run --with fastapi --with uvicorn --with pillow --with transformers \
  python scripts/qwen35_image_api.py

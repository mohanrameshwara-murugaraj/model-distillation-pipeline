#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after a completed training run. Install vllm into
# a separate CUDA-compatible environment; requires a GPU and model access.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/train/banking77-qlora-r8/final-adapter}"
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
  echo "Missing adapter at $ADAPTER_PATH. Train first." >&2
  exit 1
fi
vllm serve "$BASE_MODEL" \
  --host 127.0.0.1 --port 8000 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "banking77-lora=$ADAPTER_PATH" \
  --max-model-len 2048

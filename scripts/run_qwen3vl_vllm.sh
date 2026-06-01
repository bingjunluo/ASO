#!/usr/bin/env bash
set -euo pipefail

: "${VICTIM_CUDA_VISIBLE_DEVICES:?Set VICTIM_CUDA_VISIBLE_DEVICES to an idle GPU for Qwen3-VL, preferably outside the 7-GPU ASO training node.}"
export CUDA_VISIBLE_DEVICES="$VICTIM_CUDA_VISIBLE_DEVICES"

MODEL=${VICTIM_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
SERVED_NAME=${VICTIM_SERVED_MODEL_NAME:-qwen3-vl-8b}
HOST=${VICTIM_HOST:-0.0.0.0}
PORT=${VICTIM_PORT:-8021}

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --dtype "${VICTIM_DTYPE:-bfloat16}" \
  --max-model-len "${VICTIM_MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${VICTIM_GPU_MEMORY_UTILIZATION:-0.90}"

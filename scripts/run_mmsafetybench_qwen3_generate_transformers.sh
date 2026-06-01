#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${VICTIM_CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export VICTIM_MODEL_PATH=${VICTIM_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}

MANIFEST=${MANIFEST:-data/MM-SafetyBench/mmsafetybench_sd_typo_baseline.json}
RESPONSES=${RESPONSES:-outputs/baseline/mmsafetybench_sd_typo_qwen3-vl-8b_responses.jsonl}

python tools/generate_mmsafetybench_responses_transformers.py \
  --manifest "$MANIFEST" \
  --output "$RESPONSES" \
  --model "$VICTIM_MODEL_PATH" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${RESUME:+--resume}

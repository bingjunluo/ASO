#!/usr/bin/env bash
set -euo pipefail

export VICTIM_API_BASE_URL=${VICTIM_API_BASE_URL:-http://localhost:8021/v1}
export VICTIM_API_KEY=${VICTIM_API_KEY:-EMPTY}
export VICTIM_SERVED_MODEL_NAME=${VICTIM_SERVED_MODEL_NAME:-qwen3-vl-8b}

MANIFEST=${MANIFEST:-data/MM-SafetyBench/mmsafetybench_sd_typo_baseline.json}
RESPONSES=${RESPONSES:-outputs/baseline/mmsafetybench_sd_typo_qwen3-vl-8b_responses.jsonl}

python tools/generate_mmsafetybench_responses.py \
  --manifest "$MANIFEST" \
  --output "$RESPONSES" \
  --victim-base-url "$VICTIM_API_BASE_URL" \
  --victim-api-key "$VICTIM_API_KEY" \
  --victim-model "$VICTIM_SERVED_MODEL_NAME" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${RESUME:+--resume}

#!/usr/bin/env bash
set -euo pipefail

export HARM_BENCH_API_URL=${HARM_BENCH_API_URL:-http://localhost:5000}

RESPONSES=${RESPONSES:-outputs/baseline/mmsafetybench_sd_typo_qwen3-vl-8b_responses.jsonl}
OUTPUT=${OUTPUT:-outputs/baseline/mmsafetybench_sd_typo_qwen3-vl-8b_harmbench.jsonl}

python tools/score_mmsafetybench_responses.py \
  --responses "$RESPONSES" \
  --output "$OUTPUT" \
  --harmbench-url "$HARM_BENCH_API_URL" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${RESUME:+--resume}

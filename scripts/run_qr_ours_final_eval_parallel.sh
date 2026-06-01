#!/usr/bin/env bash
set -euo pipefail

ASO_CODE=${ASO_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RUN_ID=${RUN_ID:-qr_ours_final_eval_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$ASO_CODE/logs/$RUN_ID}
OUT_DIR=${OUT_DIR:-$ASO_CODE/outputs/qr_ours_final_eval/$RUN_ID}
: "${SAVE_DIR:?Set SAVE_DIR to the ASO training output directory containing final_records_rank*.jsonl.}"
BASELINE_MANIFEST=${BASELINE_MANIFEST:-${ASO_DATA_ROOT:-$ASO_CODE/data}/MM-SafetyBench/MM-SafetyBench_imgs_clean/mmsafetybench_sd_typo.json}
MANIFEST=${MANIFEST:-$OUT_DIR/final_records_eval_manifest.json}
SHARD_DIR=${SHARD_DIR:-$RUN_DIR/shards}
NUM_SHARDS=${NUM_SHARDS:-8}
QWEN_PORTS=${QWEN_PORTS:-8021,8022,8023,8024,8025,8026,8027,8028}
EXTRA_QWEN_GPUS=${EXTRA_QWEN_GPUS:-0,1,2,3,4,5,6}
EXTRA_QWEN_PORTS=${EXTRA_QWEN_PORTS:-8022,8023,8024,8025,8026,8027,8028}
START_EXTRA_QWEN=${START_EXTRA_QWEN:-1}
AUTO_STOP_EXTRA_QWEN=${AUTO_STOP_EXTRA_QWEN:-1}
VICTIM_MODEL_PATH=${VICTIM_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
VICTIM_SERVED_MODEL_NAME=${VICTIM_SERVED_MODEL_NAME:-qwen3-vl-8b}
VICTIM_ENV_ACTIVATE=${VICTIM_ENV_ACTIVATE:-}
HARM_BENCH_API_URL=${HARM_BENCH_API_URL:-http://127.0.0.1:5000}

mkdir -p "$RUN_DIR/qwen_logs" "$RUN_DIR/worker_logs" "$OUT_DIR" "$SHARD_DIR"
cd "$ASO_CODE"

{
  echo "RUN_ID=$RUN_ID"
  echo "UTC_START=$(date -u '+%F %T')"
  echo "ASO_CODE=$ASO_CODE"
  echo "RUN_DIR=$RUN_DIR"
  echo "OUT_DIR=$OUT_DIR"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "BASELINE_MANIFEST=$BASELINE_MANIFEST"
  echo "MANIFEST=$MANIFEST"
  echo "SHARD_DIR=$SHARD_DIR"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "QWEN_PORTS=$QWEN_PORTS"
  echo "EXTRA_QWEN_GPUS=$EXTRA_QWEN_GPUS"
  echo "EXTRA_QWEN_PORTS=$EXTRA_QWEN_PORTS"
  echo "START_EXTRA_QWEN=$START_EXTRA_QWEN"
  echo "AUTO_STOP_EXTRA_QWEN=$AUTO_STOP_EXTRA_QWEN"
  echo "VICTIM_MODEL_PATH=$VICTIM_MODEL_PATH"
  echo "VICTIM_SERVED_MODEL_NAME=$VICTIM_SERVED_MODEL_NAME"
  echo "HARM_BENCH_API_URL=$HARM_BENCH_API_URL"
} > "$RUN_DIR/run_info.txt"

python3 tools/build_final_eval_manifest.py \
  --code-dir "$ASO_CODE" \
  --save-dir "$SAVE_DIR" \
  --baseline-manifest "$BASELINE_MANIFEST" \
  --manifest-output "$MANIFEST" \
  --shard-dir "$SHARD_DIR" \
  --num-shards "$NUM_SHARDS" | tee "$RUN_DIR/build_manifest.log"

activate_victim_env() {
  if [ -n "$VICTIM_ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$VICTIM_ENV_ACTIVATE"
  fi
}

wait_for_port() {
  local port=$1
  local label=$2
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "$label ready on port $port"
      return 0
    fi
    sleep 5
  done
  echo "$label did not become ready on port $port" >&2
  return 1
}

stop_extra_qwen() {
  [ "$AUTO_STOP_EXTRA_QWEN" = "1" ] || return 0
  IFS=',' read -r -a stop_ports <<< "$EXTRA_QWEN_PORTS"
  for port in "${stop_ports[@]}"; do
    mapfile -t pids < <(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" || true)
    for pid in "${pids[@]}"; do
      cmd=$(ps -p "$pid" -o cmd= || true)
      if [[ "$cmd" == *"$VICTIM_MODEL_PATH"* ]]; then
        kill "$pid" >/dev/null 2>&1 || true
      fi
    done
  done
}
trap stop_extra_qwen EXIT

IFS=',' read -r -a ports <<< "$QWEN_PORTS"
if [ "${#ports[@]}" -ne "$NUM_SHARDS" ]; then
  echo "QWEN_PORTS must contain NUM_SHARDS entries" >&2
  exit 2
fi

wait_for_port "${ports[0]}" "qwen_existing"

if [ "$START_EXTRA_QWEN" = "1" ]; then
  IFS=',' read -r -a extra_gpus <<< "$EXTRA_QWEN_GPUS"
  IFS=',' read -r -a extra_ports <<< "$EXTRA_QWEN_PORTS"
  if [ "${#extra_gpus[@]}" -ne "${#extra_ports[@]}" ]; then
    echo "EXTRA_QWEN_GPUS and EXTRA_QWEN_PORTS must have the same length" >&2
    exit 2
  fi

  for idx in "${!extra_gpus[@]}"; do
    gpu=${extra_gpus[$idx]}
    port=${extra_ports[$idx]}
    log="$RUN_DIR/qwen_logs/qwen_gpu${gpu}_port${port}.log"
    pid_file="$RUN_DIR/qwen_logs/qwen_gpu${gpu}_port${port}.pid"
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "Qwen already ready on port $port; not starting a duplicate."
      continue
    fi
    (
      activate_victim_env
      export CUDA_VISIBLE_DEVICES=$gpu
      export VICTIM_CUDA_VISIBLE_DEVICES=$gpu
      export VICTIM_MODEL_PATH="$VICTIM_MODEL_PATH"
      export VICTIM_SERVED_MODEL_NAME="$VICTIM_SERVED_MODEL_NAME"
      export VICTIM_HOST=0.0.0.0
      export VICTIM_PORT=$port
      exec bash scripts/run_qwen3vl_vllm.sh
    ) > "$log" 2>&1 &
    echo "$!" > "$pid_file"
    echo "Started Qwen GPU $gpu port $port pid $(cat "$pid_file")"
  done

  for port in "${extra_ports[@]}"; do
    wait_for_port "$port" "qwen_extra"
  done
fi

for idx in $(seq 0 $((NUM_SHARDS - 1))); do
  port=${ports[$idx]}
  shard="$SHARD_DIR/shard_$(printf '%02d' "$idx").json"
  output="$OUT_DIR/qr_ours_final_eval_${RUN_ID}_shard_$(printf '%02d' "$idx")_port_${port}.jsonl"
  log="$RUN_DIR/worker_logs/shard_$(printf '%02d' "$idx")_port_${port}.log"
  pid_file="$RUN_DIR/worker_logs/shard_$(printf '%02d' "$idx")_port_${port}.pid"
  {
    echo "SHARD_${idx}=$shard"
    echo "OUTPUT_${idx}=$output"
    echo "PORT_${idx}=$port"
  } >> "$RUN_DIR/run_info.txt"
  (
    activate_victim_env
    python tools/eval_mmsafetybench_baseline.py \
      --manifest "$shard" \
      --output "$output" \
      --victim-base-url "http://127.0.0.1:${port}/v1" \
      --victim-api-key EMPTY \
      --victim-model "$VICTIM_SERVED_MODEL_NAME" \
      --harmbench-url "$HARM_BENCH_API_URL" \
      --timeout 240 \
      --resume
  ) > "$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "Started eval shard $idx port $port pid $(cat "$pid_file")"
done

worker_failed=0
for pid_file in "$RUN_DIR"/worker_logs/shard_*_port_*.pid; do
  pid=$(cat "$pid_file")
  if ! wait "$pid"; then
    echo "Worker failed: $pid_file pid=$pid" >&2
    worker_failed=1
  fi
done

python3 tools/summarize_harmbench_jsonl.py \
  --output-dir "$OUT_DIR" \
  --summary "$OUT_DIR/summary.json" \
  --expected 1680 | tee "$RUN_DIR/summary.log"

date -u '+UTC_END=%F %T' >> "$RUN_DIR/run_info.txt"
exit "$worker_failed"

#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:-complex_logprob_flux_kontext_mmsafetybench_qr_qwen3_vl_dynamic}
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
IFS=',' read -r -a TRAIN_GPU_IDS <<< "$TRAIN_CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
export ASO_NUM_TRAIN_GPUS=${ASO_NUM_TRAIN_GPUS:-${#TRAIN_GPU_IDS[@]}}
export ASO_DATA_ROOT=${ASO_DATA_ROOT:-"$PWD/data"}
export FLUX_KONTEXT_MODEL=${FLUX_KONTEXT_MODEL:-black-forest-labs/FLUX.1-Kontext-dev}
export HARM_BENCH_API_URL=${HARM_BENCH_API_URL:-http://localhost:5000}
export VICTIM_API_BASE_URL=${VICTIM_API_BASE_URL:-http://localhost:8021/v1}
export WANDB_MODE=${WANDB_MODE:-offline}
export ASO_TARGET_SAMPLES_PER_EPOCH=${ASO_TARGET_SAMPLES_PER_EPOCH:-48}

accelerate launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
  --num_processes="$ASO_NUM_TRAIN_GPUS" \
  --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
  scripts/train_flux_kontext_search_dynamic.py \
  --config "config/grpo_logprob.py:${CONFIG_NAME}"

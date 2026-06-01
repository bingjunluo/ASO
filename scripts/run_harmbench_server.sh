#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${HARM_BENCH_CUDA_VISIBLE_DEVICES:-7}
export HARM_BENCH_HOST=${HARM_BENCH_HOST:-0.0.0.0}
export HARM_BENCH_PORT=${HARM_BENCH_PORT:-5000}
export HARM_BENCH_MODEL=${HARM_BENCH_MODEL:-cais/HarmBench-Llama-2-13b-cls}
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

python api_server.py

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data_refactor/trumans}"
PROMPT_SUBDIR="${PROMPT_SUBDIR:-prompts}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-prompts_rewrite}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data_processing/video_caption/prompt/rewrite.txt}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ENGINE="${ENGINE:-auto}"
NUM_SPLITS="${NUM_SPLITS:-8}"
MAX_RETRY_COUNT="${MAX_RETRY_COUNT:-10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

if [[ -n "${SLURM_ARRAY_TASK_ID+set}" ]]; then
  ARRAY_ARGS=(--array_index "${SLURM_ARRAY_TASK_ID}" --num_splits "${NUM_SPLITS}")
else
  ARRAY_ARGS=()
fi

TP_ARGS=()
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  TP_ARGS=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
fi

python3 "${SCRIPT_DIR}/caption_rewrite.py" \
  --root_dir "${ROOT_DIR}" \
  --prompt_subdir "${PROMPT_SUBDIR}" \
  --output_folder_name "${OUTPUT_FOLDER_NAME}" \
  --prompt_file "${PROMPT_FILE}" \
  --model_name "${MODEL_NAME}" \
  --engine "${ENGINE}" \
  --max_retry_count "${MAX_RETRY_COUNT}" \
  --prefix '"rewritten description": ' \
  --answer_template 'your rewritten description here' \
  --temperature "${TEMPERATURE}" \
  --max_tokens "${MAX_TOKENS}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  "${TP_ARGS[@]}" \
  --skip_existing \
  "${ARRAY_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data/taste_rob}"
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2-40B-AWQ}"
ARRAY_INDEX="${ARRAY_INDEX:-0}"
NUM_SPLITS="${NUM_SPLITS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

TP_ARGS=()
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  TP_ARGS=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
fi

python "${PROJECT_ROOT}/data_processing/video_caption/internvl2_video_recaptioning.py" \
  --root_dir "${ROOT_DIR}" \
  --video_type egocentric \
  --video_folder_name videos \
  --output_folder_name prompts \
  --input_prompt_file "${PROJECT_ROOT}/data_processing/video_caption/prompt/caption_ego.txt" \
  --model_path "${MODEL_PATH}" \
  --num_sampled_frames 16 \
  --frame_sample_method uniform \
  --batch_size 1 \
  --num_workers 4 \
  --dataset_type taste_rob \
  --save_format txt \
  --use_third_person_context \
  --third_prompt_dirname prompts_aux \
  --array_index "${ARRAY_INDEX}" \
  --num_splits "${NUM_SPLITS}" \
  --split_by video \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  "${TP_ARGS[@]}" \
  --skip_existing

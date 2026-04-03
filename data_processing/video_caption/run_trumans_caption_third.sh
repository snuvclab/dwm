#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data/trumans/ego_render_fov90}"
ACTIONS_ROOT="${ACTIONS_ROOT:-${PROJECT_ROOT}/data/trumans/Actions}"
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2-40B-AWQ}"
SCENE_FILTER_FILE="${SCENE_FILTER_FILE:-}"
ARRAY_INDEX="${ARRAY_INDEX:-0}"
NUM_SPLITS="${NUM_SPLITS:-1}"
ATTACH_ACTION_HINTS="${ATTACH_ACTION_HINTS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"

SCENE_FILTER_ARG=()
if [[ -n "${SCENE_FILTER_FILE}" ]]; then
  SCENE_FILTER_ARG=(--scene_filter_file "${SCENE_FILTER_FILE}")
fi

TP_ARGS=()
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  TP_ARGS=(--tensor_parallel_size "${TENSOR_PARALLEL_SIZE}")
fi

python "${PROJECT_ROOT}/data_processing/video_caption/internvl2_video_recaptioning.py" \
  --root_dir "${ROOT_DIR}" \
  --video_type third_person \
  --video_folder_name videos_third \
  --output_folder_name prompts_aux \
  --input_prompt_file "${PROJECT_ROOT}/data_processing/video_caption/prompt/caption_third.txt" \
  --model_path "${MODEL_PATH}" \
  --num_sampled_frames 16 \
  --frame_sample_method uniform \
  --batch_size 1 \
  --num_workers 4 \
  --dataset_type trumans \
  --save_format json \
  --array_index "${ARRAY_INDEX}" \
  --num_splits "${NUM_SPLITS}" \
  --split_by action \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  "${TP_ARGS[@]}" \
  --skip_existing \
  "${SCENE_FILTER_ARG[@]}"

if [[ "${ATTACH_ACTION_HINTS}" == "1" ]]; then
  python "${PROJECT_ROOT}/data_processing/trumans/build_prompts_aux_trumans.py" \
    --root_dir "${ROOT_DIR}" \
    --actions_root "${ACTIONS_ROOT}" \
    --third_prompt_dirname prompts_aux \
    --third_video_dirname videos_third \
    --clip_length 49 \
    --clip_stride 25 \
    --frame_skip 3 \
    --skip_existing \
    "${SCENE_FILTER_ARG[@]}"
fi

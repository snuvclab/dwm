#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data_refactor/trumans}"
CAPTION_BACKEND="${CAPTION_BACKEND:-internvl2}"
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2-40B-AWQ}"
MODEL_NAME="${MODEL_NAME:-Salesforce/blip-image-captioning-base}"
BLIP_MAX_NEW_TOKENS="${BLIP_MAX_NEW_TOKENS:-32}"
SCENE_FILTER_FILE="${SCENE_FILTER_FILE:-}"
ARRAY_INDEX="${ARRAY_INDEX:-0}"
NUM_SPLITS="${NUM_SPLITS:-1}"
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

case "${CAPTION_BACKEND}" in
  internvl2)
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
      --dataset_type trumans \
      --save_format txt \
      --use_third_person_context \
      --third_prompt_dirname prompts_aux \
      --array_index "${ARRAY_INDEX}" \
      --num_splits "${NUM_SPLITS}" \
      --split_by action \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      "${TP_ARGS[@]}" \
      --skip_existing \
      "${SCENE_FILTER_ARG[@]}"
    ;;
  blip)
    python "${PROJECT_ROOT}/data_processing/video_caption/caption_videos_blip.py" \
      --root_dir "${ROOT_DIR}" \
      --dataset_type trumans \
      --video_folder_name videos \
      --output_folder_name prompts \
      --array_index "${ARRAY_INDEX}" \
      --num_splits "${NUM_SPLITS}" \
      --split_by action \
      --num_sampled_frames 3 \
      --model_name "${MODEL_NAME}" \
      --max_new_tokens "${BLIP_MAX_NEW_TOKENS}" \
      --skip_existing \
      "${SCENE_FILTER_ARG[@]}"
    ;;
  *)
    echo "Unsupported CAPTION_BACKEND: ${CAPTION_BACKEND}" >&2
    exit 1
    ;;
esac

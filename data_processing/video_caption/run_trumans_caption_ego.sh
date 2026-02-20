#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}/data/trumans/ego_render_fov90}"
MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL2-40B-AWQ}"
SCENE_FILTER_FILE="${SCENE_FILTER_FILE:-}"
ARRAY_INDEX="${ARRAY_INDEX:-0}"
NUM_SPLITS="${NUM_SPLITS:-1}"

SCENE_FILTER_ARG=()
if [[ -n "${SCENE_FILTER_FILE}" ]]; then
  SCENE_FILTER_ARG=(--scene_filter_file "${SCENE_FILTER_FILE}")
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
  --dataset_type trumans \
  --save_format txt \
  --use_third_person_context \
  --third_prompt_dirname prompts_aux \
  --array_index "${ARRAY_INDEX}" \
  --num_splits "${NUM_SPLITS}" \
  --split_by action \
  --skip_existing \
  "${SCENE_FILTER_ARG[@]}"

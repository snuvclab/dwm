#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INFER_SCRIPT="${REPO_ROOT}/training/cogvideox/inference.py"
DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/hf_release/cogvideox_hand_concat_lora_ego_fun_rewrite_checkpoint-8000"
DEFAULT_EXPERIMENT_CONFIG="${DEFAULT_CHECKPOINT_PATH}/dwm_cogvideox_5b_lora.yaml"
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
DEFAULT_DATASET_FILE="${REPO_ROOT}/data/dataset_files/realworld_selected_48.txt"
DEFAULT_PROMPT_SUBDIR="prompts_rewrite"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/outputs_infer/$(basename "${DEFAULT_CHECKPOINT_PATH}")/$(basename "${DEFAULT_DATASET_FILE%.txt}")"

source ~/.bashrc || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda initialization script" >&2
  exit 1
fi

conda activate dwm
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

HAS_CHECKPOINT_PATH=false
HAS_EXPERIMENT_CONFIG=false
HAS_DATA_ROOT=false
HAS_DATASET_FILE=false
HAS_OUTPUT_DIR=false
HAS_PROMPT_SUBDIR=false
HAS_DATASET_OUTPUT_LAYOUT=false

for arg in "$@"; do
  case "$arg" in
    --checkpoint_path) HAS_CHECKPOINT_PATH=true ;;
    --experiment_config) HAS_EXPERIMENT_CONFIG=true ;;
    --data_root) HAS_DATA_ROOT=true ;;
    --dataset_file) HAS_DATASET_FILE=true ;;
    --output_dir) HAS_OUTPUT_DIR=true ;;
    --prompt_subdir) HAS_PROMPT_SUBDIR=true ;;
    --dataset_output_layout) HAS_DATASET_OUTPUT_LAYOUT=true ;;
  esac
done

ARGS=("$@")
if [[ "${HAS_DATASET_OUTPUT_LAYOUT}" == false ]]; then
  ARGS=(--dataset_output_layout per_sample "${ARGS[@]}")
fi
if [[ "${HAS_PROMPT_SUBDIR}" == false ]]; then
  ARGS=(--prompt_subdir "${DEFAULT_PROMPT_SUBDIR}" "${ARGS[@]}")
fi
if [[ "${HAS_OUTPUT_DIR}" == false ]]; then
  ARGS=(--output_dir "${DEFAULT_OUTPUT_DIR}" "${ARGS[@]}")
fi
if [[ "${HAS_DATASET_FILE}" == false ]]; then
  ARGS=(--dataset_file "${DEFAULT_DATASET_FILE}" "${ARGS[@]}")
fi
if [[ "${HAS_DATA_ROOT}" == false ]]; then
  ARGS=(--data_root "${DEFAULT_DATA_ROOT}" "${ARGS[@]}")
fi
if [[ "${HAS_EXPERIMENT_CONFIG}" == false ]]; then
  ARGS=(--experiment_config "${DEFAULT_EXPERIMENT_CONFIG}" "${ARGS[@]}")
fi
if [[ "${HAS_CHECKPOINT_PATH}" == false ]]; then
  ARGS=(--checkpoint_path "${DEFAULT_CHECKPOINT_PATH}" "${ARGS[@]}")
fi

python "${INFER_SCRIPT}" "${ARGS[@]}"

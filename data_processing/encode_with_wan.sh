#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENCODER_SCRIPT="${REPO_ROOT}/data_processing/encode_with_wan.py"

DEBUG_MODE=false
LOCAL_MODE=false
DATASET_TYPE=""
DATA_ROOT=""
PROMPT_SUBDIR="prompts_rewrite"
WAN_VERSION="2.1"
MODALITIES=()
EXTRA_ARGS=()
SKIP_EXISTING=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --local)
      LOCAL_MODE=true
      shift
      ;;
    --dataset_type)
      DATASET_TYPE="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --prompt_subdir)
      PROMPT_SUBDIR="$2"
      shift 2
      ;;
    --wan_version)
      WAN_VERSION="$2"
      shift 2
      ;;
    --modalities)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODALITIES+=("$1")
        shift
      done
      ;;
    --skip_existing)
      SKIP_EXISTING=true
      shift
      ;;
    --overwrite)
      SKIP_EXISTING=false
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATASET_TYPE" || -z "$DATA_ROOT" || ${#MODALITIES[@]} -eq 0 ]]; then
  echo "Usage: $0 --dataset_type {trumans|taste_rob} --data_root <path> --modalities videos static_videos hand_videos prompts" >&2
  exit 1
fi

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
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

COMMON_ARGS=(
  --dataset_type "$DATASET_TYPE"
  --data_root "$DATA_ROOT"
  --modalities "${MODALITIES[@]}"
  --prompt_subdir "$PROMPT_SUBDIR"
  --wan_version "$WAN_VERSION"
  "${EXTRA_ARGS[@]}"
)

if [[ "$SKIP_EXISTING" == true ]]; then
  COMMON_ARGS+=(--skip_existing)
fi

if [[ "$DEBUG_MODE" == true ]]; then
  python "$ENCODER_SCRIPT" "${COMMON_ARGS[@]}" --debug
  exit $?
fi

if [[ "$LOCAL_MODE" == true ]]; then
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  mkdir -p out
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python "$ENCODER_SCRIPT" "${COMMON_ARGS[@]}" --rank "$i" --world_size "$NUM_GPUS" > "out/encode_wan_rank_${i}.log" 2>&1 &
  done
  wait
  exit $?
fi

python "$ENCODER_SCRIPT" "${COMMON_ARGS[@]}"

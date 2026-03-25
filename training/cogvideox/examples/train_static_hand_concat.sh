#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/training/cogvideox/configs/examples/static_hand_concat_lora_rewrite.yaml"
TRAIN_SCRIPT="${REPO_ROOT}/training/cogvideox/train_dwm_cogvideox.py"

DEBUG_MODE=false
SLURM_TEST_MODE=false
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --slurm_test)
      SLURM_TEST_MODE=true
      shift
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

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

COMMON_OVERRIDES=()
if [[ "$DEBUG_MODE" == true ]]; then
  COMMON_OVERRIDES+=(
    "training.max_train_steps=10"
    "training.batch_size=1"
    "data.dataloader_num_workers=0"
    "data.max_validation_videos=0"
  )
fi
if [[ "$SLURM_TEST_MODE" == true ]]; then
  COMMON_OVERRIDES+=(
    "training.max_train_steps=10"
    "data.dataloader_num_workers=0"
    "data.max_validation_videos=0"
  )
fi
COMMON_OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

if [[ "$DEBUG_MODE" == true ]]; then
  python "$TRAIN_SCRIPT" \
    --experiment_config "$CONFIG_PATH" \
    --mode debug \
    --override "${COMMON_OVERRIDES[@]}"
  exit $?
fi

if [[ "$SLURM_TEST_MODE" == true ]]; then
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$TRAIN_SCRIPT" \
    --experiment_config "$CONFIG_PATH" \
    --mode slurm_test \
    --override "${COMMON_OVERRIDES[@]}"
  exit $?
fi

python "$TRAIN_SCRIPT" \
  --experiment_config "$CONFIG_PATH" \
  --mode batch \
  --override "${COMMON_OVERRIDES[@]}"

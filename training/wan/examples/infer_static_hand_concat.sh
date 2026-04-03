#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/training/wan/configs/examples/dwm_wan_14b_lora.yaml"
INFER_SCRIPT="${REPO_ROOT}/training/wan/inference.py"

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

HAS_EXPERIMENT_CONFIG=false
for arg in "$@"; do
  if [[ "$arg" == "--experiment_config" ]]; then
    HAS_EXPERIMENT_CONFIG=true
    break
  fi
done

ARGS=("$@")
if [[ "$HAS_EXPERIMENT_CONFIG" == false ]]; then
  ARGS=(--experiment_config "$CONFIG_PATH" "${ARGS[@]}")
fi

python "$INFER_SCRIPT" "${ARGS[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=dwm-cogvideox-rw48
#SBATCH --partition=background
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-7
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
HELPER_SCRIPT="${REPO_ROOT}/training/cogvideox/examples/infer_realworld.sh"

cd "${REPO_ROOT}"
bash "${HELPER_SCRIPT}" --chunk_id "${SLURM_ARRAY_TASK_ID}" --num_chunks 8 "$@"

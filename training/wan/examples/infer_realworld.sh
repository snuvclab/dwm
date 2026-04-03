#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INFER_SCRIPT="${REPO_ROOT}/training/wan/inference.py"
DEFAULT_CHECKPOINT_PATH="${REPO_ROOT}/hf_release/wan2.1_14b_fun_inp_hand_concat_lora_checkpoint-5000"
DEFAULT_EXPERIMENT_CONFIG="${DEFAULT_CHECKPOINT_PATH}/dwm_wan_14b_lora.yaml"
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
DEFAULT_DATASET_FILE="${REPO_ROOT}/data/dataset_files/realworld_selected_48.txt"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/outputs_infer"
DEFAULT_PROMPT_SUBDIR="prompts_rewrite"

usage() {
  cat <<'EOF'
Usage:
  bash training/wan/examples/infer_realworld.sh [options]

Options:
  --chunk_id N             Chunk id. Default: SLURM_ARRAY_TASK_ID or 0.
  --num_chunks N           Number of chunks. Default: SLURM_ARRAY_TASK_COUNT or 1.
  --checkpoint_path PATH   Checkpoint directory.
  --experiment_config PATH Experiment YAML path.
  --data_root PATH         Data root directory.
  --dataset_file PATH      Dataset file to split. Default: data/dataset_files/realworld_selected_48.txt
  --output_root PATH       Root output directory. Default: outputs_infer
  --prompt_subdir NAME     Prompt subdirectory. Default: prompts_rewrite
  --help                   Show this message.

All remaining arguments are forwarded to inference.py.
If multiple GPUs are visible through CUDA_VISIBLE_DEVICES, the chunk is processed
with torchrun and sharded automatically across those GPUs.
EOF
}

CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_CHUNKS="${SLURM_ARRAY_TASK_COUNT:-1}"
CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH}"
EXPERIMENT_CONFIG="${DEFAULT_EXPERIMENT_CONFIG}"
DATA_ROOT="${DEFAULT_DATA_ROOT}"
DATASET_FILE="${DEFAULT_DATASET_FILE}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
PROMPT_SUBDIR="${DEFAULT_PROMPT_SUBDIR}"
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunk_id)
      CHUNK_ID="$2"
      shift 2
      ;;
    --num_chunks)
      NUM_CHUNKS="$2"
      shift 2
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --experiment_config)
      EXPERIMENT_CONFIG="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --dataset_file)
      DATASET_FILE="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --prompt_subdir)
      PROMPT_SUBDIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CHUNK_ID}" ]]; then
  echo "Missing --chunk_id and SLURM_ARRAY_TASK_ID is not set." >&2
  exit 1
fi

if ! [[ "${CHUNK_ID}" =~ ^[0-9]+$ ]]; then
  echo "Chunk id must be a non-negative integer: ${CHUNK_ID}" >&2
  exit 1
fi

if ! [[ "${NUM_CHUNKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_chunks must be a positive integer: ${NUM_CHUNKS}" >&2
  exit 1
fi

if (( CHUNK_ID >= NUM_CHUNKS )); then
  echo "chunk_id must satisfy 0 <= chunk_id < num_chunks: ${CHUNK_ID} vs ${NUM_CHUNKS}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_FILE}" ]]; then
  echo "Dataset file not found: ${DATASET_FILE}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EXPERIMENT_CONFIG}" ]]; then
  echo "Experiment config not found: ${EXPERIMENT_CONFIG}" >&2
  exit 1
fi

echo "[INFO] chunk=${CHUNK_ID}/${NUM_CHUNKS}" >&2
echo "[INFO] dataset_file=${DATASET_FILE}" >&2

CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH%/}")"
DATASET_NAME="$(basename "${DATASET_FILE%.txt}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${CHECKPOINT_NAME}/${DATASET_NAME}"
mkdir -p "${OUTPUT_DIR}"

VISIBLE_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a RAW_VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  for GPU_ID in "${RAW_VISIBLE_GPUS[@]}"; do
    if [[ -n "${GPU_ID}" ]]; then
      VISIBLE_GPUS+=("${GPU_ID}")
    fi
  done
fi
if (( ${#VISIBLE_GPUS[@]} == 0 )); then
  VISIBLE_GPUS=("0")
fi

echo "[INFO] visible_gpus=${VISIBLE_GPUS[*]}" >&2
echo "[INFO] output_dir=${OUTPUT_DIR}" >&2

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda initialization script" >&2
  exit 1
fi

conda activate dwm
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

COMMON_ARGS=(
  "${INFER_SCRIPT}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --experiment_config "${EXPERIMENT_CONFIG}"
  --data_root "${DATA_ROOT}"
  --dataset_file "${DATASET_FILE}"
  --prompt_subdir "${PROMPT_SUBDIR}"
  --output_dir "${OUTPUT_DIR}"
  --chunk_id "${CHUNK_ID}"
  --num_chunks "${NUM_CHUNKS}"
)

if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${PASSTHROUGH_ARGS[@]}")
fi

if (( ${#VISIBLE_GPUS[@]} > 1 )); then
  torchrun --standalone --nnodes=1 --nproc_per_node="${#VISIBLE_GPUS[@]}" "${COMMON_ARGS[@]}"
else
  python "${COMMON_ARGS[@]}"
fi

#!/bin/bash
set -euo pipefail

DEBUG_MODE=false
LOCAL_MODE=false
DATASET_TYPE=""
DATA_ROOT=""
MODALITIES=""
PROMPT_SUBDIR="prompts_rewrite"
MODEL_ID="THUDM/CogVideoX-5b"
SKIP_EXISTING="--skip_existing"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --modalities)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODALITIES="$MODALITIES $1"
                shift
            done
            ;;
        --prompt_subdir)
            PROMPT_SUBDIR="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING="--skip_existing"
            shift
            ;;
        --overwrite)
            SKIP_EXISTING=""
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$DATASET_TYPE" ] || [ -z "$MODALITIES" ]; then
    echo "Usage: $0 --dataset_type {trumans|taste_rob} --modalities {videos|static_videos|hand_videos|prompts}... [--data_root PATH]"
    exit 1
fi

MODALITIES=$(echo "$MODALITIES" | xargs)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source ~/.bashrc || true

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
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
    --modalities $MODALITIES
    --prompt_subdir "$PROMPT_SUBDIR"
    --model_id "$MODEL_ID"
)

if [ -n "$DATA_ROOT" ]; then
    COMMON_ARGS+=(--data_root "$DATA_ROOT")
fi

if [ "$DEBUG_MODE" = true ]; then
    python "$REPO_ROOT/data_processing/encode_with_cogvideox.py" \
        "${COMMON_ARGS[@]}" \
        --rank 0 \
        --world_size 1 \
        --debug \
        $SKIP_EXISTING \
        $EXTRA_ARGS
    exit $?
fi

if [ "$LOCAL_MODE" = true ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i python "$REPO_ROOT/data_processing/encode_with_cogvideox.py" \
            "${COMMON_ARGS[@]}" \
            --rank $i \
            --world_size $NUM_GPUS \
            $SKIP_EXISTING \
            $EXTRA_ARGS &
    done
    wait
    exit $?
fi

python "$REPO_ROOT/data_processing/encode_with_cogvideox.py" \
    "${COMMON_ARGS[@]}" \
    --rank 0 \
    --world_size 1 \
    $SKIP_EXISTING \
    $EXTRA_ARGS

#!/bin/bash
#SBATCH --job-name=cvx_fun_lora
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=train
#SBATCH --nodelist=compute-st-kait-gpu-2
#SBATCH --comment="Training CogVideoX Fun with static scene and hand pose concatenation"
#SBATCH --output=out/cvX_fun_lora_%j.out
#SBATCH --error=out/cvX_fun_lora_%j.err

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate world_model

# Set PYTHONPATH to include the current directory for module imports
export PYTHONPATH="${PYTHONPATH}:/fsx/byungjun_vclab/world_model"

# LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/trumans_static_pose_all/test_selected_50.txt"
# LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/taste_rob/rob_test_selected_48.txt"
# LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/trumans_static_pose_all/temp.txt"
LIST_FILE="data/dataset_files/trumans_static_pose_all/test_selected_50.txt"
CHECKPOINT_NUM=${1:-4000}  # Get checkpoint number from $1, default to 4000
CHECKPOINT="outputs/251115/trumans_fun_static_hand_concat_sft_from_i2v_slurm_5175/checkpoint-${CHECKPOINT_NUM}"
CONFIG="outputs/251115/trumans_fun_static_hand_concat_sft_from_i2v_slurm_5175/trumans_fun_static_hand_concat_sft_from_i2v.yaml"
DEBUG=false

echo "[INFO] Using checkpoint: ${CHECKPOINT}"


# Optional configuration
NUM_SPLITS=4
PYTHON_BIN=python

# Get GPUs from CUDA_VISIBLE_DEVICES
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[WARNING] CUDA_VISIBLE_DEVICES not set, assuming 0,1,2,3"
    GPUS=(0 1 2 3)
else
    # Parse CUDA_VISIBLE_DEVICES (e.g., "4,5,6,7" -> GPUS=(4 5 6 7))
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    echo "[INFO] Detected ${#GPUS[@]} GPU(s) from CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

if [ "$2" = "--debug" ]; then
    DEBUG=true
    NUM_SPLITS=1
    # Use only the first GPU from CUDA_VISIBLE_DEVICES
    echo "[DEBUG] Debug mode enabled: using first GPU from CUDA_VISIBLE_DEVICES (${GPUS[0]}) without splitting."
fi

# Verify input file
if [ ! -f "$LIST_FILE" ]; then
    echo "Error: input list file not found: $LIST_FILE"
    exit 1
fi


# -----------------
# Split input file
# -----------------
if [ "$DEBUG" = true ]; then
    echo "[INFO] Debug mode: skipping file split."
    PARTS=("$LIST_FILE")
else
    echo "[INFO] Splitting $LIST_FILE into $NUM_SPLITS parts..."
    split -n l/$NUM_SPLITS "$LIST_FILE" "${LIST_FILE}.part"

    PARTS=($(ls ${LIST_FILE}.part*))
    if [ ${#PARTS[@]} -ne $NUM_SPLITS ]; then
        echo "[ERROR] Split failed (expected $NUM_SPLITS parts)."
        exit 1
    fi
fi

# -----------------
# Launch processes
# -----------------
echo "[INFO] Launching ${NUM_SPLITS} worker(s)..."

for i in $(seq 0 $((NUM_SPLITS-1))); do
    PART=${PARTS[$i]}
    GPU=${GPUS[$i]}
    echo "[LAUNCH] GPU ${GPU} -> ${PART}"

    if [ "$DEBUG" = true ]; then
        # Debug mode: run in foreground (no &)
        CUDA_VISIBLE_DEVICES=${GPU} \
        $PYTHON_BIN training/cogvideox_static_pose/inference_sft_i2v.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --video_list "${PART}" \
        --output_dir "${CHECKPOINT}/eval_for_quant"
    else
        # Normal mode: run in background
        CUDA_VISIBLE_DEVICES=${GPU} \
        $PYTHON_BIN training/cogvideox_static_pose/inference_sft_i2v.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --video_list "${PART}" \
        --output_dir "${CHECKPOINT}/eval_for_quant" &
    fi

done

if [ "$DEBUG" != true ]; then
    echo "[INFO] All processes started in background."
    
    # -----------------
    # Wait for all jobs
    # -----------------
    wait
    echo "[INFO] All GPU jobs finished."
else
    echo "[INFO] Debug mode: job completed (foreground execution)."
fi
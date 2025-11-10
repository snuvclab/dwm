#!/bin/bash
#SBATCH --job-name=cvx_fun_lora
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=jhb_vclab
#SBATCH --comment="Training CogVideoX Fun with static scene and hand pose concatenation"
#SBATCH --output=out/cvX_fun_lora_%j.out
#SBATCH --error=out/cvX_fun_lora_%j.err

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate world_model

LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/trumans_static_pose_all/test_selected_50.txt"
# LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/taste_rob/rob_test_selected_48.txt"
# LIST_FILE="/virtual_lab/jhb_vclab/world_model/data/dataset_files/trumans_static_pose_all/temp.txt"
LORA_PATH="/virtual_lab/jhb_vclab/world_model/outputs/251105/trumans_fun_static_hand_concat_lora_new_prompt_slurm_3783/checkpoint-6000/" # Ours
# LORA_PATH="/virtual_lab/jhb_vclab/world_model/outputs/251027/trumans_fun_static_lora_slurm_76734/checkpoint-3000/" # CogVideoX Fun LoRA


# Optional configuration
NUM_SPLITS=4
GPUS=(0 1 2 3)
PYTHON_BIN=python3

# Verify input file
if [ ! -f "$LIST_FILE" ]; then
    echo "Error: input list file not found: $LIST_FILE"
    exit 1
fi


# -----------------
# Split input file
# -----------------
echo "[INFO] Splitting $LIST_FILE into $NUM_SPLITS parts..."
split -n l/$NUM_SPLITS "$LIST_FILE" "${LIST_FILE}.part"

PARTS=($(ls ${LIST_FILE}.part*))
if [ ${#PARTS[@]} -ne $NUM_SPLITS ]; then
    echo "[ERROR] Split failed (expected $NUM_SPLITS parts)."
    exit 1
fi

# -----------------
# Launch processes
# -----------------
echo "[INFO] Launching 4 parallel workers..."

for i in $(seq 0 $((NUM_SPLITS-1))); do
    PART=${PARTS[$i]}
    GPU=${GPUS[$i]}
    echo "[LAUNCH] GPU ${GPU} -> ${PART}"

    CUDA_VISIBLE_DEVICES=${GPU} \
    $PYTHON_BIN training/cogvideox_static_pose/inference_lora.py \
    --txt_path "${PART}" \
    --lora_path "${LORA_PATH}" &

done

echo "[INFO] All processes started in background."

# -----------------
# Wait for all jobs
# -----------------
wait
echo "[INFO] All GPU jobs finished."
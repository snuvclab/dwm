#!/bin/bash
#SBATCH --job-name=world_model_lab_00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=batch
#SBATCH --output=out/world_model_lab_00_%j.out
#SBATCH --error=out/world_model_lab_00_%j.err
#SBATCH --comment="Training World Model on Lab 00 Dataset camera pose estimation"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate world_model

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16

GPU_IDS="0,1,2,3"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATE="1e-4"
LR_SCHEDULE="cosine_with_restarts"
OPTIMIZER="adamw"
MAX_TRAIN_STEPS="5000"

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_4.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/virtual_lab/jhb_vclab/world_model/data"
DATASET_FILE="dataset_files/captured.txt"
VAL_FILE="dataset_files/captured_test.txt"

# Launch experiments with different hyperparameters

output_dir="outputs/aether_lab00-01/"

cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/aether/aether_lora.py \
    --pretrained_cogvideox_name_or_path THUDM/CogVideoX-5b-I2V \
    --pretrained_aether_name_or_path AetherWorldModel/AetherV1 \
    --data_root $DATA_ROOT \
    --dataset_file $DATASET_FILE \
    --height_buckets 480 \
    --width_buckets 720 \
    --frame_buckets 49 \
    --dataloader_num_workers 8 \
    --pin_memory \
    --validation_set $VAL_FILE \
    --num_validation_videos 1 \
    --validation_steps 250 \
    --seed 42 \
    --rank 128 \
    --lora_alpha 128 \
    --mixed_precision bf16 \
    --output_dir $output_dir \
    --max_num_frames 49 \
    --train_batch_size 1 \
    --max_train_steps $MAX_TRAIN_STEPS \
    --checkpointing_steps 1000 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULE \
    --lr_warmup_steps 0 \
    --lr_num_cycles 1 \
    --enable_slicing \
    --enable_tiling \
    --optimizer $OPTIMIZER \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --allow_tf32 \
    --report_to wandb \
    --load_tensors \
    --nccl_timeout 1800 \
    --use_empty_prompts"

        
echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
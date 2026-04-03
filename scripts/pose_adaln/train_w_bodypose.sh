#!/bin/bash
#SBATCH --job-name=trumans_aether_training
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=batch
#SBATCH --output=out/trumans_aether_%j.out
#SBATCH --error=out/trumans_aether_%j.err

# source ~/.bashrc
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate world_model

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16

GPU_IDS="0,1"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATE="1e-4"
LR_SCHEDULE="cosine_with_restarts"
OPTIMIZER="adamw"
MAX_TRAIN_STEPS="10000"
BATCH_SIZE=4

# Multi-GPU training with DeepSpeed (2 GPUs)
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_2.yaml"

# Absolute path to where the processed Trumans data is located
DATA_ROOT="/virtual_lab/jhb_vclab/world_model/data/trumans/ego_render_new_processed"
CAPTION_COLUMN="prompts_train.txt"
VIDEO_COLUMN="videos_train.txt"

# Output directory for training
output_dir="outputs/trumans_full_aether_w_bodypose_fixed/"

# Launch training with Aether on Trumans dataset
# cmd="python training/aether/aether_smplx_lora.py \
cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/aether/aether_smplx_lora.py \
    --pretrained_cogvideox_name_or_path THUDM/CogVideoX-5b-I2V \
    --pretrained_aether_name_or_path AetherWorldModel/AetherV1 \
    --data_root $DATA_ROOT \
    --caption_column $CAPTION_COLUMN \
    --video_column $VIDEO_COLUMN \
    --height_buckets 480 \
    --width_buckets 720 \
    --frame_buckets 49 \
    --dataloader_num_workers 8 \
    --pin_memory \
    --validation_set $DATA_ROOT/videos_val.txt \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_steps 250 \
    --seed 42 \
    --rank 64 \
    --lora_alpha 64 \
    --mixed_precision bf16 \
    --output_dir $output_dir \
    --max_num_frames 49 \
    --train_batch_size $BATCH_SIZE \
    --max_train_steps $MAX_TRAIN_STEPS \
    --checkpointing_steps 500 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULE \
    --lr_warmup_steps 400 \
    --lr_num_cycles 1 \
    --enable_slicing \
    --enable_tiling \
    --noised_image_dropout 0.05 \
    --optimizer $OPTIMIZER \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --allow_tf32 \
    --report_to wandb \
    --load_tensors \
    --nccl_timeout 1800 \
    --enable_pose_conditioning \
    --use_empty_prompts"

echo "===== Training Aether on Trumans Dataset ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Data root: $DATA_ROOT"
echo "Output directory: $output_dir"
echo "Learning rate: $LEARNING_RATE"
echo "Max train steps: $MAX_TRAIN_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: $GPU_IDS"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# Check if split files exist
if [ ! -f "$DATA_ROOT/videos_train.txt" ]; then
    echo "Error: Training split file not found: $DATA_ROOT/videos_train.txt"
    echo "Please run the split creation script first:"
    echo "python training/aether/create_trumans_splits.py --data_root $DATA_ROOT --output_dir $DATA_ROOT"
    exit 1
fi

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n" 
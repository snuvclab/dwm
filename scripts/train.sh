#!/bin/bash
#SBATCH --job-name=world_model_lab_00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=batch
#SBATCH --output=out/world_model_lab_00_%j.out
#SBATCH --error=out/world_model_lab_00_%j.err

# source ~/.bashrc
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate tavid

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
MAX_TRAIN_STEPS="5000"

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/processed"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"

# Launch experiments with different hyperparameters

output_dir="outputs/lab_00_naive/"

cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox/cogvideox_image_to_video_lora.py \
    --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
    --data_root $DATA_ROOT \
    --caption_column $CAPTION_COLUMN \
    --video_column $VIDEO_COLUMN \
    --id_token BW_STYLE \
    --height_buckets 480 \
    --width_buckets 720 \
    --frame_buckets 49 \
    --dataloader_num_workers 8 \
    --pin_memory \
    --validation_prompt \"An egocentric view of a person walking around the SNUVCLAB office.:::An egocentric view of a person walking around the SNUVCLAB office.:::An egocentric view of a person walking around the SNUVCLAB office.:::An egocentric view of a person walking around the SNUVCLAB office.:::An egocentric view of a person walking around the SNUVCLAB office.:::An egocentric view of a person walking around the SNUVCLAB office.\" \
    --validation_images \"/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/00264.png:::/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/00434.png:::/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/00574.png:::/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/00934.png:::/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/01614.png:::/virtual_lab/jhb_vclab/taeksoo/data/world_model/lab_00/validation/02699.png\"
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_steps 500 \
    --seed 42 \
    --rank 128 \
    --lora_alpha 64 \
    --mixed_precision bf16 \
    --output_dir $output_dir \
    --max_num_frames 49 \
    --train_batch_size 2 \
    --max_train_steps $MAX_TRAIN_STEPS \
    --checkpointing_steps 1000 \
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
    --nccl_timeout 1800"
        
echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
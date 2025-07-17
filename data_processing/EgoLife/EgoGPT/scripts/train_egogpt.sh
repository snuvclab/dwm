run_name=EgoGPT-7B
# export WANDB_API_KEY=Your_WANDB_API_KEY
# wandb login $WANDB_API_KEY
# export WANDB_PROJECT=EgoGPT
# export WANDB_NAME=$run_name
# wandb online

# Replace with downloaded dataset path
DATA_PATH=./datasets/EgoLife_Depersonalized_EgoIT.json
# Replace with downloaded model path
MODEL_PATH=lmms-lab/llava-onevision-qwen2-7b-ov
# Replace with downloaded speech projector path
SPEECH_PROJECTOR_PATH=./pretrained/speech_projector_ov.bin
# Replace with downloaded speech encoder path
SPEECH_ENCODER_PATH=./pretrained/large-v3.pt

torchrun --nproc_per_node=8 \
    --master_port=10043 \
    egogpt/train/train_audio.py \
    --deepspeed ./scripts/zero3.json \
    --run_name $run_name \
    --model_name_or_path $MODEL_PATH \
    --version qwen_1_5 \
    --data_path $DATA_PATH \
    --pretrain_speech_projector $SPEECH_PROJECTOR_PATH \
    --bf16 True \
    --output_dir ./checkpoints/$run_name \
    --sample_independently True \
    --vision_tower google/siglip-so400m-patch14-384 \
    --speech_encoder  $SPEECH_ENCODER_PATH \
    --speech_encoder_type whisper \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --frames_upbound 30 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --min_lr_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
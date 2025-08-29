#!/bin/bash

# Batch processing script for Trumans dataset
# Processes all scenes and action sequences under a base directory

export PYTHONPATH=/virtual_lab/jhb_vclab/byungjun_vclab/world_model:$PYTHONPATH

# Signal handling for graceful shutdown
cleanup() {
    echo ""
    echo "===== Received interrupt signal, cleaning up... ====="
    
    # Set a timeout for cleanup operations
    TIMEOUT=10
    
    # Kill the main process group first
    if [ ! -z "$TORCHRUN_PID" ]; then
        echo "Killing main process group: $TORCHRUN_PID"
        kill -TERM $TORCHRUN_PID 2>/dev/null
        
        # Wait for graceful termination with timeout
        for i in $(seq 1 $TIMEOUT); do
            if ! kill -0 $TORCHRUN_PID 2>/dev/null; then
                echo "Process terminated gracefully"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 $TORCHRUN_PID 2>/dev/null; then
            echo "Force killing process..."
            kill -KILL $TORCHRUN_PID 2>/dev/null
        fi
    fi
    
    # Kill any Python processes related to our script
    echo "Killing Python processes..."
    pkill -f "prepare_dataset_trumans_batch.py" 2>/dev/null
    pkill -f "prepare_dataset_trumans.py" 2>/dev/null
    
    # Kill any torchrun processes
    echo "Killing torchrun processes..."
    pkill -f "torchrun" 2>/dev/null
    
    # Kill any remaining Python processes that might be stuck
    echo "Killing any remaining Python processes..."
    pkill -f "python.*training/aether" 2>/dev/null
    
    # Force kill any defunct processes
    echo "Cleaning up defunct processes..."
    kill -9 $(ps aux | grep -E "(defunct|Z)" | grep -v grep | awk '{print $2}') 2>/dev/null 2>/dev/null
    
    echo "Cleanup completed"
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

MODEL_ID="THUDM/CogVideoX-5b-I2V"
NUM_GPUS=1  # Number of GPUs to use for processing

# Set to 1 for easier process management (easier to kill)
# Set to 2 or more for faster processing
USE_SINGLE_GPU=true  # Set to true for easier Ctrl+C handling

if [ "$USE_SINGLE_GPU" = true ]; then
    NUM_GPUS=1
    echo "Using single GPU mode for easier process management"
fi

# Base directory containing scene folders
# DATA_ROOT="/virtual_lab/jhb_vclab/byungjun_vclab/world_model/data/trumans/250712_sample"
DATA_ROOT="/virtual_lab/jhb_vclab/byungjun_vclab/world_model/data/trumans/ego_render_fov90"

# Optional: Process only a specific scene
# Leave empty to process all scenes, or specify a scene name to process only that scene
# Example: SCENE_NAME="0a7618195-4647-889b-a726747201"
SCENE_NAME="0a761819-05d1-4647-889b-a726747201b1-copy"

# Configuration
HEIGHT_BUCKETS="480"
WIDTH_BUCKETS="720"
FRAME_BUCKETS="49"
MAX_NUM_FRAMES="49"
MAX_SEQUENCE_LENGTH=226
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=fp32

# Sequences directory name under each action directory
SEQUENCES_DIR="sequences"

# Model type for predefined file type combinations
# Options: "aether", "cogvideox_pose", "custom"
# - "aether": All file types for Aether training
# - "cogvideox_pose": Only file types needed for CogVideoX pose training
# - "custom": Use CHECK_FILE_TYPES below for manual specification
MODEL_TYPE="aether"

# Example configurations:
# For Aether training:
# MODEL_TYPE="aether"
# 
# For CogVideoX pose training:
# MODEL_TYPE="cogvideox_pose"
# 
# For custom selection:
# MODEL_TYPE="custom"
# CHECK_FILE_TYPES="videos video_latents hand_videos hand_video_latents"

# File types to check for skip_existing (only used when MODEL_TYPE="custom")
# Leave empty to check all file types, or specify only the ones you care about
# CHECK_FILE_TYPES="human_motions"  # Example: only check videos and human_motions
# CHECK_FILE_TYPES="prompts prompt_embeds"

# Command to run batch processing
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU mode - easier to kill with Ctrl+C
    CMD="python data_processing/trumans/prepare_dataset_trumans_batch.py \
        --model_id $MODEL_ID \
        --data_root $DATA_ROOT \
        --caption_column prompts.txt \
        --video_column videos.txt \
        --height_buckets $HEIGHT_BUCKETS \
        --width_buckets $WIDTH_BUCKETS \
        --frame_buckets $FRAME_BUCKETS \
        --max_num_frames $MAX_NUM_FRAMES \
        --max_sequence_length $MAX_SEQUENCE_LENGTH \
        --target_fps $TARGET_FPS \
        --batch_size $BATCH_SIZE \
        --dtype $DTYPE \
        --sequences_dir $SEQUENCES_DIR \
        --model_type $MODEL_TYPE"
    
    # Add scene_name argument if specified
    if [ ! -z "$SCENE_NAME" ]; then
        CMD="$CMD --scene_filter $SCENE_NAME"
    fi

    # Add full processing flags for Aether model type
    if [ "$MODEL_TYPE" = "aether" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_image_latents --save_prompt_embeds"
    fi

    if [ "$MODEL_TYPE" = "cogvideox_pose" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_prompt_embeds"
    fi
    
    # Add full processing flags only when using custom model type and CHECK_FILE_TYPES is not specified
    if [ "$MODEL_TYPE" = "custom" ] && [ -z "$CHECK_FILE_TYPES" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_image_latents --save_prompt_embeds"
    fi
    
    # Add check_file_types if using custom model type and specified
    if [ "$MODEL_TYPE" = "custom" ] && [ ! -z "$CHECK_FILE_TYPES" ]; then
        CMD="$CMD --check_file_types $CHECK_FILE_TYPES"
    fi
else
    # Multi-GPU mode with torchrun
    CMD="torchrun --nproc_per_node=$NUM_GPUS data_processing/trumans/prepare_dataset_trumans_batch.py \
        --model_id $MODEL_ID \
        --data_root $DATA_ROOT \
        --video_column videos.txt \
        --caption_column prompts.txt \
        --height_buckets $HEIGHT_BUCKETS \
        --width_buckets $WIDTH_BUCKETS \
        --frame_buckets $FRAME_BUCKETS \
        --max_num_frames $MAX_NUM_FRAMES \
        --max_sequence_length $MAX_SEQUENCE_LENGTH \
        --target_fps $TARGET_FPS \
        --batch_size $BATCH_SIZE \
        --dtype $DTYPE \
        --sequences_dir $SEQUENCES_DIR \
        --model_type $MODEL_TYPE"
    
    # Add scene filter if specified
    if [ ! -z "$SCENE_NAME" ]; then
        CMD="$CMD --scene_filter $SCENE_NAME"
    fi

    # Add full processing flags for Aether model type
    if [ "$MODEL_TYPE" = "aether" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_image_latents --save_prompt_embeds"
    fi

    if [ "$MODEL_TYPE" = "cogvideox_pose" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_prompt_embeds"
    fi
    
    # Add full processing flags only when using custom model type and CHECK_FILE_TYPES is not specified
    if [ "$MODEL_TYPE" = "custom" ] && [ -z "$CHECK_FILE_TYPES" ]; then
        CMD="$CMD --save_latents_and_embeddings --save_image_latents --save_prompt_embeds"
    fi
    
    # Add check_file_types if using custom model type and specified
    if [ "$MODEL_TYPE" = "custom" ] && [ ! -z "$CHECK_FILE_TYPES" ]; then
        CMD="$CMD --check_file_types $CHECK_FILE_TYPES"
    fi
fi

echo "===== Running batch processing for Trumans dataset ====="
echo "Data root: $DATA_ROOT"
if [ ! -z "$SCENE_NAME" ]; then
    echo "Scene filter: $SCENE_NAME (processing only this scene)"
else
    echo "Scene filter: None (processing all scenes)"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_ID"
echo "Model type: $MODEL_TYPE"
echo "Number of GPUs: $NUM_GPUS"
echo "Height buckets: $HEIGHT_BUCKETS"
echo "Width buckets: $WIDTH_BUCKETS"
echo "Frame buckets: $FRAME_BUCKETS"
echo "Max frames: $MAX_NUM_FRAMES"
echo "Target FPS: $TARGET_FPS"
echo "Batch size: $BATCH_SIZE"
echo "Data type: $DTYPE"
if [ "$MODEL_TYPE" = "custom" ]; then
    if [ ! -z "$CHECK_FILE_TYPES" ]; then
        echo "Check file types: $CHECK_FILE_TYPES (custom selection)"
    else
        echo "Check file types: ALL (custom mode, slower but thorough)"
    fi
else
    echo "Check file types: Predefined for $MODEL_TYPE model"
fi
echo "Note: Use Ctrl+C to gracefully stop the processing"
echo ""

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# # Create output directory
# mkdir -p "$OUTPUT_DIR"

echo "Command: $CMD"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "===== Running in SINGLE GPU mode (foreground execution) ====="
    echo "Command: $CMD"
    echo ""
    
    $CMD
    EXIT_CODE=$?
else
    echo "===== Running in MULTI GPU mode (background execution) ====="
    echo "Command: $CMD"
    echo ""

    eval $CMD &
    TORCHRUN_PID=$!
    echo "Main process started with PID: $TORCHRUN_PID"
    echo "Use Ctrl+C to stop the processing"
    echo ""

    wait $TORCHRUN_PID
    EXIT_CODE=$?
fi

echo ""
echo "===== Batch processing completed ====="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Processing completed successfully"
    echo "All processed files saved to: $OUTPUT_DIR"
else
    echo "❌ Processing failed with exit code: $EXIT_CODE"
    echo "Check the output above for error details"
fi 
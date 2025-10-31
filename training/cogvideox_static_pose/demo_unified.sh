#!/bin/bash

# Unified CogVideoX Inference Demo Script
# Supports all pipeline types: concat, adapter, adaln, i2v
# Supports both single GPU and multi-GPU inference

set -e  # Exit on any error

# Default configuration
PIPELINE_TYPE=""
EXPERIMENT_CONFIG=""
CHECKPOINT_PATH=""
DATASET_FILE=""
DATA_ROOT="./data"
OUTPUT_DIR=""
EVAL_SUBFOLDER="eval"
NUM_GPUS=1
GPU_IDS=""

# Inference parameters
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=6.0
USE_DYNAMIC_CFG=true
HEIGHT=480
WIDTH=720
NUM_FRAMES=49
FPS=8
SEED=42

# Processing options
MAX_BATCH_SIZE=""
USE_EMPTY_PROMPTS=false
COMPUTE_METRICS=true
SAVE_COMPARISON_VIDEOS=true
VERBOSE=true
SUFFIX=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --checkpoint_path PATH   Path to trained checkpoint directory"
    echo "  --dataset_file PATH      Path to dataset file containing video paths"
    echo ""
    echo "Pipeline configuration (optional - auto-detected from checkpoint if not provided):"
    echo "  --experiment_config PATH Path to experiment YAML config file"
    echo "  --pipeline_type TYPE     Pipeline type (e.g., cogvideox_fun_static_to_video_raymap_pose_concat)"
    echo ""
    echo "Optional arguments:"
    echo "  --output_dir PATH        Path to save outputs (default: {checkpoint_path}/{eval_subfolder})"
    echo "  --eval_subfolder NAME    Subfolder name under checkpoint_path (default: eval)"
    echo "  --data_root PATH         Data root directory (default: ./data)"
    echo "  --num_gpus N             Number of GPUs for distributed inference (default: 1)"
    echo "  --gpu_ids IDS            Comma-separated GPU IDs (e.g., '0,1,2,3')"
    echo "  --num_inference_steps N  Number of inference steps (default: 50)"
    echo "  --guidance_scale F       Guidance scale (default: 6.0)"
    echo "  --height H               Video height (default: 480)"
    echo "  --width W                Video width (default: 720)"
    echo "  --num_frames N           Number of frames (default: 49)"
    echo "  --fps N                  Frames per second (default: 8)"
    echo "  --max_batch_size N       Maximum number of videos to process"
    echo "  --seed N                 Random seed (default: 42)"
    echo "  --use_empty_prompts      Use empty prompts instead of provided ones"
    echo "  --no_metrics             Disable video quality metrics computation"
    echo "  --no_comparison_videos   Disable comparison video saving"
    echo "  --verbose                Enable verbose logging"
    echo "  --suffix SUFFIX          Suffix to append to output filenames (e.g., '_no_prompt', '_cfg7.5')"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Using experiment config (recommended)"
    echo "  $0 --experiment_config ./exps/250912/trumans_fun_static_hand_adapter_lora.yaml \\"
    echo "     --checkpoint_path ./outputs/250912/trumans_fun_static_hand_adapter_lora/checkpoint-1000/ \\"
    echo "     --dataset_file ./data/dataset_files/trumans_static_pose_0a7618/test.txt \\"
    echo "     --output_dir ./outputs/inference/fun_static_hand_adapter_test \\"
    echo "     --suffix '_cfg7.5'"
    echo ""
    echo "  # Single GPU inference with explicit pipeline type"
    echo "  $0 --pipeline_type cogvideox_pose_concat \\"
    echo "     --checkpoint_path ./outputs/250901/trumans_concat_static_hand_slurm/checkpoint-1000/ \\"
    echo "     --dataset_file ./data/dataset_files/trumans_static_pose_0a7618/test.txt \\"
    echo "     --output_dir ./outputs/inference/concat_test"
    echo ""
    echo "  # Multi-GPU inference with AdaLN pipeline"
    echo "  $0 --pipeline_type cogvideox_pose_adaln \\"
    echo "     --checkpoint_path ./outputs/250903/trumans_adaln_slurm/checkpoint-1000/ \\"
    echo "     --dataset_file ./data/dataset_files/trumans_static_pose_0a7618/test.txt \\"
    echo "     --output_dir ./outputs/inference/adaln_test \\"
    echo "     --num_gpus 4 --gpu_ids '0,1,2,3'"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_config)
            EXPERIMENT_CONFIG="$2"
            shift 2
            ;;
        --pipeline_type)
            PIPELINE_TYPE="$2"
            shift 2
            ;;
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset_file)
            DATASET_FILE="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_subfolder)
            EVAL_SUBFOLDER="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --num_inference_steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance_scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --max_batch_size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --use_empty_prompts)
            USE_EMPTY_PROMPTS=true
            shift
            ;;
        --no_metrics)
            COMPUTE_METRICS=false
            shift
            ;;
        --no_comparison_videos)
            SAVE_COMPARISON_VIDEOS=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: --checkpoint_path is required"
    usage
    exit 1
fi

if [[ -z "$DATASET_FILE" ]]; then
    echo "Error: --dataset_file is required"
    usage
    exit 1
fi

# Validate pipeline configuration (both are optional - inference script will auto-detect)
if [[ -n "$EXPERIMENT_CONFIG" && -n "$PIPELINE_TYPE" ]]; then
    echo "⚠️  Warning: Both --experiment_config and --pipeline_type provided. Using --experiment_config and ignoring --pipeline_type"
    PIPELINE_TYPE=""
fi

if [[ -z "$EXPERIMENT_CONFIG" && -z "$PIPELINE_TYPE" ]]; then
    echo "🔧 Note: No --experiment_config or --pipeline_type provided. The inference script will auto-detect the config from the checkpoint directory."
fi

# Check if experiment config file exists
if [[ -n "$EXPERIMENT_CONFIG" && ! -f "$EXPERIMENT_CONFIG" ]]; then
    echo "Error: Experiment config file does not exist: $EXPERIMENT_CONFIG"
    exit 1
fi

# Set default output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$CHECKPOINT_PATH/$EVAL_SUBFOLDER"
    echo "🔧 Auto-generated output directory: $OUTPUT_DIR"
fi

# Validate pipeline type (only if not using experiment config)
if [[ -n "$PIPELINE_TYPE" ]]; then
    case $PIPELINE_TYPE in
        cogvideox_pose_concat|cogvideox_pose_adapter|cogvideox_pose_adaln|cogvideox_i2v|cogvideox_static_to_video|cogvideox_static_to_video_pose_concat|cogvideox_fun_static_to_video_pose_concat|cogvideox_fun_static_to_video_cross|cogvideox_fun_static_to_video_cross_pose_adapter|cogvideox_fun_static_to_video_pose_adapter)
            ;;
        *)
            echo "Error: Invalid pipeline type: $PIPELINE_TYPE"
            echo "Valid options: cogvideox_pose_concat, cogvideox_pose_adapter, cogvideox_pose_adaln, cogvideox_i2v, cogvideox_static_to_video, cogvideox_static_to_video_pose_concat, cogvideox_fun_static_to_video_pose_concat, cogvideox_fun_static_to_video_cross, cogvideox_fun_static_to_video_cross_pose_adapter, cogvideox_fun_static_to_video_pose_adapter"
            exit 1
            ;;
    esac
fi

# Check if checkpoint path exists
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Check if dataset file exists
if [[ ! -f "$DATASET_FILE" ]]; then
    echo "Error: Dataset file does not exist: $DATASET_FILE"
    exit 1
fi

# Check if data root exists
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "🎯 CogVideoX Unified Inference Demo"
echo "=================================="
if [[ -n "$EXPERIMENT_CONFIG" ]]; then
    echo "Experiment Config: $EXPERIMENT_CONFIG"
    echo "Pipeline Type: (from config)"
else
    echo "Pipeline Type: $PIPELINE_TYPE"
fi
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset File: $DATASET_FILE"
echo "Data Root: $DATA_ROOT"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
if [[ -n "$GPU_IDS" ]]; then
    echo "GPU IDs: $GPU_IDS"
fi
echo "Inference Steps: $NUM_INFERENCE_STEPS"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "Video Size: ${HEIGHT}x${WIDTH}"
echo "Number of Frames: $NUM_FRAMES"
echo "FPS: $FPS"
echo "Seed: $SEED"
if [[ -n "$MAX_BATCH_SIZE" ]]; then
    echo "Max Batch Size: $MAX_BATCH_SIZE"
fi
echo "Use Empty Prompts: $USE_EMPTY_PROMPTS"
echo "Compute Metrics: $COMPUTE_METRICS"
echo "Save Comparison Videos: $SAVE_COMPARISON_VIDEOS"
echo "Verbose: $VERBOSE"
if [[ -n "$SUFFIX" ]]; then
    echo "Filename Suffix: $SUFFIX"
fi
echo "=================================="

# Build command
if [[ $NUM_GPUS -eq 1 ]]; then
    # Single GPU inference
    echo "🚀 Starting single GPU inference..."
    
    CMD="python -m ipdb -c continue training/cogvideox_static_pose/inference_unified.py"
    
    # Set GPU ID if specified
    if [[ -n "$GPU_IDS" ]]; then
        # Extract first GPU ID from comma-separated list
        GPU_ID=$(echo "$GPU_IDS" | cut -d',' -f1)
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "🔧 Using GPU ID: $GPU_ID"
    fi
    
    # Add pipeline configuration (optional - will auto-detect if not provided)
    if [[ -n "$EXPERIMENT_CONFIG" ]]; then
        CMD="$CMD --experiment_config $EXPERIMENT_CONFIG"
    elif [[ -n "$PIPELINE_TYPE" ]]; then
        CMD="$CMD --pipeline_type $PIPELINE_TYPE"
    fi
    
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
    CMD="$CMD --dataset_file $DATASET_FILE"
    CMD="$CMD --data_root $DATA_ROOT"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --eval_subfolder $EVAL_SUBFOLDER"
    CMD="$CMD --num_inference_steps $NUM_INFERENCE_STEPS"
    CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
    CMD="$CMD --height $HEIGHT"
    CMD="$CMD --width $WIDTH"
    CMD="$CMD --num_frames $NUM_FRAMES"
    CMD="$CMD --fps $FPS"
    CMD="$CMD --seed $SEED"
    
    if [[ "$USE_DYNAMIC_CFG" == "true" ]]; then
        CMD="$CMD --use_dynamic_cfg"
    fi
    
    if [[ "$USE_EMPTY_PROMPTS" == "true" ]]; then
        CMD="$CMD --use_empty_prompts"
    fi
    
    if [[ "$COMPUTE_METRICS" == "true" ]]; then
        CMD="$CMD --compute_metrics"
    fi
    
    if [[ "$SAVE_COMPARISON_VIDEOS" == "true" ]]; then
        CMD="$CMD --save_comparison_videos"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMD="$CMD --verbose"
    fi
    
    if [[ -n "$MAX_BATCH_SIZE" ]]; then
        CMD="$CMD --max_batch_size $MAX_BATCH_SIZE"
    fi
    
    if [[ -n "$SUFFIX" ]]; then
        CMD="$CMD --suffix '$SUFFIX'"
    fi
    
else
    # Multi-GPU inference
    echo "🚀 Starting multi-GPU inference on $NUM_GPUS GPUs..."
    
    CMD="python training/cogvideox_static_pose/inference_distributed.py"
    
    # Add pipeline configuration (optional - will auto-detect if not provided)
    if [[ -n "$EXPERIMENT_CONFIG" ]]; then
        CMD="$CMD --experiment_config $EXPERIMENT_CONFIG"
    elif [[ -n "$PIPELINE_TYPE" ]]; then
        CMD="$CMD --pipeline_type $PIPELINE_TYPE"
    fi
    
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH"
    CMD="$CMD --dataset_file $DATASET_FILE"
    CMD="$CMD --data_root $DATA_ROOT"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --num_gpus $NUM_GPUS"
    CMD="$CMD --num_inference_steps $NUM_INFERENCE_STEPS"
    CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
    CMD="$CMD --height $HEIGHT"
    CMD="$CMD --width $WIDTH"
    CMD="$CMD --num_frames $NUM_FRAMES"
    CMD="$CMD --fps $FPS"
    CMD="$CMD --seed $SEED"
    
    if [[ -n "$GPU_IDS" ]]; then
        CMD="$CMD --gpu_ids $GPU_IDS"
    fi
    
    if [[ "$USE_DYNAMIC_CFG" == "true" ]]; then
        CMD="$CMD --use_dynamic_cfg"
    fi
    
    if [[ "$USE_EMPTY_PROMPTS" == "true" ]]; then
        CMD="$CMD --use_empty_prompts"
    fi
    
    if [[ "$COMPUTE_METRICS" == "true" ]]; then
        CMD="$CMD --compute_metrics"
    fi
    
    if [[ "$SAVE_COMPARISON_VIDEOS" == "true" ]]; then
        CMD="$CMD --save_comparison_videos"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        CMD="$CMD --verbose"
    fi
    
    if [[ -n "$MAX_BATCH_SIZE" ]]; then
        CMD="$CMD --max_batch_size $MAX_BATCH_SIZE"
    fi
    
    if [[ -n "$SUFFIX" ]]; then
        CMD="$CMD --suffix '$SUFFIX'"
    fi
fi

# Print and execute command
echo "Command: $CMD"
echo ""

# Execute the command
eval $CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Inference completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    
    # Show summary if available
    if [[ $NUM_GPUS -gt 1 ]]; then
        SUMMARY_FILE="$OUTPUT_DIR/merged/merged_summary.json"
    else
        SUMMARY_FILE="$OUTPUT_DIR/batch_summary.json"
    fi
    
    if [[ -f "$SUMMARY_FILE" ]]; then
        echo ""
        echo "📊 Summary:"
        python -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    summary = json.load(f)
print(f'  Total videos: {summary.get(\"total_videos\", 0)}')
print(f'  Successful: {summary.get(\"successful\", 0)}')
print(f'  Failed: {summary.get(\"failed\", 0)}')
print(f'  Success rate: {summary.get(\"success_rate\", 0)*100:.1f}%')
"
    fi
else
    echo ""
    echo "❌ Inference failed!"
    exit 1
fi



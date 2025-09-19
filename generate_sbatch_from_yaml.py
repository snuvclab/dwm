#!/usr/bin/env python3
"""
Generate SLURM batch script from YAML configuration file.

Usage:
    python generate_sbatch_from_yaml.py <yaml_config_file> [output_script_name]

Example:
    python generate_sbatch_from_yaml.py trumans_concat_static_hand.yaml
    python generate_sbatch_from_yaml.py trumans_concat_static_hand.yaml my_training.sh
"""

import yaml
import sys
import os
from pathlib import Path
from datetime import datetime

def generate_sbatch_script(yaml_file, output_script=None):
    """Generate SLURM batch script from YAML configuration."""
    
    # Load YAML configuration
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return False
    
    # Extract date from YAML file path (e.g., exps/250901/config.yaml -> 250901)
    yaml_path = Path(yaml_file)
    date_match = None
    for part in yaml_path.parts:
        if part.isdigit() and len(part) == 6:  # YYYYMM format
            date_match = part
            break
    
    if date_match:
        print(f"📅 Auto-detected date from path: {date_match}")
    else:
        print("⚠️  No date pattern found in YAML path, using default")
        date_match = "unknown"
    
    # Extract configuration sections
    experiment = config.get('experiment', {})
    slurm = config.get('slurm', {})
    environment = config.get('environment', {})
    training = config.get('training', {})
    data = config.get('data', {})
    model = config.get('model', {})
    logging = config.get('logging', {})
    
    # Auto-update paths with extracted date and experiment name
    if date_match != "unknown":
        # Get experiment name and date from YAML
        exp_name = experiment.get('name', 'unknown_experiment')
        exp_date = experiment.get('date', 'unknown_date')
        
        # Update model output_dir: {output_dir}/{date}/{name}
        original_output_dir = model.get('output_dir', 'outputs/unknown')
        if not original_output_dir.startswith(f'outputs/{date_match}/'):
            model['output_dir'] = f'outputs/{date_match}/{exp_name}'
        
        # Update SLURM output and error paths with %j prefix for better sorting
        original_output = slurm.get('output', 'out/default_%j.out')
        original_error = slurm.get('error', 'out/default_%j.err')
        
        if not original_output.startswith(f'out/{date_match}/'):
            slurm['output'] = f'out/{date_match}/%j_{exp_name}.out'
        if not original_error.startswith(f'out/{date_match}/'):
            slurm['error'] = f'out/{date_match}/%j_{exp_name}.err'
    
    # Generate output script name if not provided
    if output_script is None:
        # Use YAML file name (without extension) as base
        yaml_stem = yaml_path.stem  # e.g., "trumans_fun_static_hand_concat_lora" from "trumans_fun_static_hand_concat_lora.yaml"
        output_script = f"{yaml_stem}_training.sh"
    
    # Ensure output script is in the same directory as YAML
    yaml_dir = Path(yaml_file).parent
    output_script = yaml_dir / output_script
    
    # Generate flexible script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm.get('job_name', 'default_job')}
#SBATCH --nodes={slurm.get('nodes', 1)}
#SBATCH --gpus={slurm.get('gpus', 2)}
#SBATCH --partition={slurm.get('partition', 'batch')}
#SBATCH --output={slurm.get('output', 'out/%j_default.out')}
#SBATCH --error={slurm.get('error', 'out/%j_default.err')}

# Generated from: {yaml_file}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Experiment: {experiment.get('name', 'unknown')}
# Description: {experiment.get('description', 'No description')}

# Signal handler for clean shutdown
cleanup() {{
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$ACCELERATE_PID" ]; then
        echo "   Killing accelerate process (PID: $ACCELERATE_PID)..."
        kill -TERM $ACCELERATE_PID 2>/dev/null
        sleep 2
        kill -KILL $ACCELERATE_PID 2>/dev/null
    fi
    
    # Kill any remaining Python processes (ULTRA SAFE: only kill direct children)
    echo "   Cleaning up child processes..."
    # Only kill processes that are direct children of this script
    if [ ! -z "$$" ]; then
        echo "   Killing child processes of this script (PID: $$)..."
        pkill -P $$ 2>/dev/null
    fi
    
    # Only kill accelerate processes that are children of this script
    if [ ! -z "$ACCELERATE_PID" ]; then
        echo "   Killing child processes of accelerate (PID: $ACCELERATE_PID)..."
        pkill -P $ACCELERATE_PID 2>/dev/null
    fi
    
    # Note: GPU processes and CUDA cache clearing removed to avoid affecting other processes
    
    echo "✅ Cleanup completed"
    exit 1
}}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Conda environment setup
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate world_model

# Set PYTHONPATH to include the current directory for module imports
export PYTHONPATH="${{PYTHONPATH}}:/virtual_lab/jhb_vclab/byungjun_vclab/world_model"

# Environment variables from YAML config
export TORCH_LOGS="{environment.get('torch_logs', '+dynamo,recompiles,graph_breaks')}"
export TORCHDYNAMO_VERBOSE={environment.get('torchdynamo_verbose', 1)}
export WANDB_MODE="{environment.get('wandb_mode', 'offline')}"
export TORCH_NCCL_ENABLE_MONITORING={environment.get('torch_nccl_enable_monitoring', 0)}
export TOKENIZERS_PARALLELISM={str(environment.get('tokenizers_parallelism', True)).lower()}
export OMP_NUM_THREADS={environment.get('omp_num_threads', 16)}

# Configuration from YAML
EXPERIMENT_CONFIG="{yaml_file}"
EXPERIMENT_NAME="{experiment.get('name', 'unknown')}"
TRAINING_MODE="{training.get('mode', 'unknown')}"
LEARNING_RATE={training.get('learning_rate', 'unknown')}
BATCH_SIZE={training.get('batch_size', 'unknown')}
MAX_TRAIN_STEPS={training.get('max_train_steps', training.get('num_epochs', 'unknown'))}
DATA_ROOT="{data.get('data_root', 'unknown')}"
DATASET_FILE="{data.get('dataset_file', 'unknown')}"
BASE_OUTPUT_DIR="{model.get('output_dir', 'unknown')}"
SLURM_JOB_NAME="{slurm.get('job_name', 'unknown')}"

# Parse command line arguments
DEBUG_MODE=false
SLURM_TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --slurm_test)
            SLURM_TEST_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug        Run in debug mode with ipdb (single GPU, batch_size=1)"
            echo "  --slurm_test   Run with accelerate launch using 2 GPUs"
            echo "  --help         Show this help message"
            echo ""
            echo "Modes:"
            echo "  1. ./$0 --debug           : Debug mode with ipdb (single GPU, batch_size=1)"
            echo "  2. ./$0 --slurm_test      : Test mode with accelerate (2 GPUs)"
            echo "  3. sbatch $0              : SLURM mode with YAML GPU count"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# GPU configuration based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug mode - always use single GPU
    NUM_GPUS=1
    GPU_IDS="0"
    ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_1.yaml"
    echo "🔧 Debug mode: Using single GPU with ipdb"
elif [ "$SLURM_TEST_MODE" = true ]; then
    # Test mode - use 2 GPUs
    NUM_GPUS=2
    GPU_IDS="0,1"
    ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_2.yaml"
    echo "🧪 SLURM test mode: Using 2 GPUs with accelerate"
else
    # SLURM mode - use GPU count from YAML
    NUM_GPUS={slurm.get('gpus', 2)}
    
    # Use SLURM allocated GPU IDs
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Use actual allocated GPUs in SLURM environment
        GPU_IDS="$CUDA_VISIBLE_DEVICES"
        echo "🔧 Using SLURM allocated GPUs: $GPU_IDS"
    else
        # Fallback: use sequential GPU IDs starting from 0
        GPU_IDS=""
        for i in $(seq 0 $((NUM_GPUS-1))); do
            if [ $i -eq 0 ]; then
                GPU_IDS="$i"
            else
                GPU_IDS="$GPU_IDS,$i"
            fi
        done
        echo "🔧 Using fallback GPU IDs: $GPU_IDS"
    fi
    
    # Accelerate config based on GPU count
    if [ $NUM_GPUS -eq 1 ]; then
        ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_2.yaml"
    elif [ $NUM_GPUS -eq 2 ]; then
        ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_2.yaml"
    elif [ $NUM_GPUS -eq 4 ]; then
        ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_4.yaml"
    elif [ $NUM_GPUS -eq 8 ]; then
        ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_8.yaml"
    else
        echo "Warning: Unsupported GPU count: $NUM_GPUS, using default config"
        ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_2.yaml"
    fi
    echo "🚀 SLURM mode: Using $NUM_GPUS GPUs from YAML config"
fi

echo "===== CogVideoX Pose Unified Training ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Training Mode: $TRAINING_MODE"
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Max Train Steps: $MAX_TRAIN_STEPS"
echo "Data Root: $DATA_ROOT"
echo "Dataset File: $DATASET_FILE"
echo "Base Output Directory: $BASE_OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"
echo "Accelerate Config: $ACCELERATE_CONFIG_FILE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT"
    exit 1
fi

# Check if dataset file exists
if [ ! -f "$DATA_ROOT/$DATASET_FILE" ]; then
    echo "Error: Dataset file not found: $DATA_ROOT/$DATASET_FILE"
    exit 1
fi

# Check if config file exists
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "Error: Experiment config file not found: $EXPERIMENT_CONFIG"
    exit 1
fi

# Create output directory (will be created by the training script based on mode)
echo "📁 Output directory will be created by training script based on mode"

# Build command
SCRIPT_NAME="training/cogvideox_static_pose/cogvideox_text_to_video_pose_sft_unified.py"

# Base command based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug mode - override batch size to 1 and validation videos to 1 for easier debugging
    cmd="python -m ipdb -c continue $SCRIPT_NAME --experiment_config $EXPERIMENT_CONFIG --mode debug --override training.batch_size=1 data.max_validation_videos=1 "
    echo "🔧 Debug mode command: $cmd"
    echo "🔧 Debug mode: Batch size overridden to 1, validation videos to 1 for easier debugging"
elif [ "$SLURM_TEST_MODE" = true ]; then
    cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS $SCRIPT_NAME --experiment_config $EXPERIMENT_CONFIG --mode slurm_test --override data.max_validation_videos=1"
    echo "🚀 Accelerate command: $cmd"
    echo "🧪 SLURM test mode: Validation videos overridden to 1 for faster testing"
else
    cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS $SCRIPT_NAME --experiment_config $EXPERIMENT_CONFIG --mode slurm"
    echo "🚀 Default command: $cmd"
fi

echo ""

# Run training
if [ "$DEBUG_MODE" = true ]; then
    echo "===== Running in DEBUG mode ====="
    echo "Using ipdb for debugging - use Ctrl+C to stop the training"
    echo ""
    
    $cmd
    EXIT_CODE=$?
else
    echo "===== Running Training ====="
    echo "Using accelerate launch for distributed training"
    echo ""
    
    echo "Command: $cmd"
    echo ""
    
    eval $cmd &
    ACCELERATE_PID=$!
    echo "Main process started with PID: $ACCELERATE_PID"
    echo "Use Ctrl+C to stop the processing"
    echo ""
    
    # Wait for the process and capture exit code
    wait $ACCELERATE_PID
    EXIT_CODE=$?
    
    # Clear the PID variable after process ends
    unset ACCELERATE_PID
fi

echo ""
echo "===== Training completed ====="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 Training Summary:"
    echo "   Experiment: $EXPERIMENT_NAME"
    echo "   Training Mode: $TRAINING_MODE"
    echo "   Learning Rate: $LEARNING_RATE"
    echo "   Batch Size: $BATCH_SIZE"
    echo "   Max Steps: $MAX_TRAIN_STEPS"
    echo "   GPUs Used: $NUM_GPUS"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check the output above for error details"
    echo ""
    echo "🔍 Debugging tips:"
    echo "   - Check if all data files exist"
    echo "   - Verify GPU availability"
    echo "   - Check memory usage"
    echo "   - Try debug mode with --debug"
fi

echo ""
echo "📁 Output files:"
echo "   - Model: Will be saved to mode-specific directory (debug/slurm_test/slurm suffix)"
echo "   - Logs: Will be saved to mode-specific directory/logs/"
echo "   - Checkpoints: Will be saved to mode-specific directory/checkpoint-*/"
echo "   - SLURM output: {slurm.get('output', 'out/%j_default.out')}"
echo "   - SLURM error: {slurm.get('error', 'out/%j_default.err')}"
"""

    # Write the script to file
    try:
        with open(output_script, 'w') as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(output_script, 0o755)
        
        print(f"✅ Generated flexible training script: {output_script}")
        print(f"📋 Script details:")
        print(f"   - Job Name: {slurm.get('job_name', 'default_job')}")
        print(f"   - Nodes: {slurm.get('nodes', 1)}")
        print(f"   - GPUs: {slurm.get('gpus', 2)}")
        print(f"   - Partition: {slurm.get('partition', 'batch')}")
        print(f"   - Output: {slurm.get('output', 'out/default_%j.out')}")
        print(f"   - Error: {slurm.get('error', 'out/default_%j.err')}")
        print(f"")
        print(f"🚀 Usage modes:")
        print(f"   1. Debug mode:     ./{output_script.name} --debug")
        print(f"   2. Test mode:      ./{output_script.name} --slurm_test")
        print(f"   3. SLURM mode:     sbatch {output_script}")
        print(f"")
        print(f"📝 To edit the script:")
        print(f"   nano {output_script}")
        
        return True
        
    except Exception as e:
        print(f"Error writing script file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_sbatch_from_yaml.py <yaml_config_file> [output_script_name]")
        print("Example: python generate_sbatch_from_yaml.py trumans_concat_static_hand.yaml")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    output_script = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(yaml_file):
        print(f"Error: YAML file not found: {yaml_file}")
        sys.exit(1)
    
    success = generate_sbatch_script(yaml_file, output_script)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


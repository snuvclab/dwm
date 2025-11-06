#!/bin/bash
#SBATCH --job-name=hawor_process
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=jhb_vclab
#SBATCH --output=out/251105/%j_hawor_process_train_split.out
#SBATCH --error=out/251105/%j_hawor_process_train_split.err

# Signal handler for clean shutdown
cleanup() {
    echo ""
    echo "🛑 Received interrupt signal. Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$PYTHON_PID" ]; then
        echo "   Killing Python process (PID: $PYTHON_PID)..."
        kill -TERM $PYTHON_PID 2>/dev/null
        sleep 2
        kill -KILL $PYTHON_PID 2>/dev/null
    fi
    
    # Kill any remaining Python processes (ULTRA SAFE: only kill direct children)
    echo "   Cleaning up child processes..."
    if [ ! -z "$$" ]; then
        echo "   Killing child processes of this script (PID: $$)..."
        pkill -P $$ 2>/dev/null
    fi
    
    # Kill any remaining Python processes that might be stuck
    echo "   Killing any remaining Python processes..."
    pkill -f "process_hawor_batch.py" 2>/dev/null
    
    # Wait a bit for cleanup to complete
    sleep 1
    
    echo "✅ Cleanup completed"
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Conda environment setup
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hawor

# Set PYTHONPATH to include the current directory for module imports
export PYTHONPATH="${PYTHONPATH}:$HOME/world_model"

# Environment variables
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16

# Configuration
TRAIN_LIST="$HOME/world_model/data/taste_rob/double/train_list.txt"
INPUT_BASE="$HOME/world_model/data/taste_rob/double_resized"
OUTPUT_BASE="$HOME/world_model/data/taste_rob"
CHECKPOINT="$HOME/world_model/data_processing/HaWoR/weights/hawor/checkpoints/hawor.ckpt"
INFILLER_WEIGHT="$HOME/world_model/data_processing/HaWoR/weights/hawor/checkpoints/infiller.pt"
MAX_WORKERS_PER_GPU=1
SKIP_EXISTING=true

# Parse command line arguments
DEBUG_MODE=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --train_list)
            TRAIN_LIST="$2"
            shift 2
            ;;
        --input_base)
            INPUT_BASE="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --infiller_weight)
            INFILLER_WEIGHT="$2"
            shift 2
            ;;
        --max_workers_per_gpu)
            MAX_WORKERS_PER_GPU="$2"
            shift 2
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug                  Run in debug mode (single GPU)"
            echo "  --test                   Run in test mode (2 GPUs)"
            echo "  --train_list PATH        Path to train_list.txt (default: $TRAIN_LIST)"
            echo "  --input_base PATH        Base directory for input videos (default: $INPUT_BASE)"
            echo "  --output_base PATH       Base directory for output videos (default: $OUTPUT_BASE)"
            echo "  --checkpoint PATH        Path to HaWoR checkpoint (default: $CHECKPOINT)"
            echo "  --infiller_weight PATH   Path to infiller weight (default: $INFILLER_WEIGHT)"
            echo "  --max_workers_per_gpu N Maximum workers per GPU (default: $MAX_WORKERS_PER_GPU)"
            echo "  --no-skip-existing       Force reprocessing even if output exists"
            echo "  --help                   Show this help message"
            echo ""
            echo "Modes:"
            echo "  1. ./$0 --debug                  : Debug mode (single GPU)"
            echo "  2. ./$0 --test                  : Test mode (2 GPUs)"
            echo "  3. sbatch $0                    : SLURM mode (4 GPUs)"
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
    echo "🔧 Debug mode: Using single GPU"
elif [ "$TEST_MODE" = true ]; then
    # Test mode - use 2 GPUs
    NUM_GPUS=2
    echo "🧪 Test mode: Using 2 GPUs"
else
    # SLURM mode - use GPU count from SLURM
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Count GPUs from CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "🔧 Using SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
    else
        # Fallback: use 4 GPUs
        NUM_GPUS=4
        echo "🔧 Using fallback: 4 GPUs"
    fi
    echo "🚀 SLURM mode: Using $NUM_GPUS GPUs"
fi

# Build GPU list
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
else
    # Build GPU list from 0 to NUM_GPUS-1
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

echo "===== HaWoR Batch Processing ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Train list: $TRAIN_LIST"
echo "Input base: $INPUT_BASE"
echo "Output base: $OUTPUT_BASE"
echo "Checkpoint: $CHECKPOINT"
echo "Infiller weight: $INFILLER_WEIGHT"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU list: $GPU_LIST"
echo "Max workers per GPU: $MAX_WORKERS_PER_GPU"
echo "Skip existing: $SKIP_EXISTING"
echo ""

# Check if train list exists
if [ ! -f "$TRAIN_LIST" ]; then
    echo "Error: Train list file does not exist: $TRAIN_LIST"
    exit 1
fi

# Check if input base exists
if [ ! -d "$INPUT_BASE" ]; then
    echo "Error: Input base directory does not exist: $INPUT_BASE"
    exit 1
fi

# Build command
SCRIPT_NAME="data_processing/HaWoR/process_hawor_batch.py"

# Base command arguments
BASE_ARGS="--train_list $TRAIN_LIST \
    --input_base $INPUT_BASE \
    --output_base $OUTPUT_BASE \
    --checkpoint $CHECKPOINT \
    --infiller_weight $INFILLER_WEIGHT \
    --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
    --gpus $GPU_LIST
"

# Add skip_existing flag
if [ "$SKIP_EXISTING" = true ]; then
    BASE_ARGS="$BASE_ARGS"
    # Note: skip_existing is default behavior, no flag needed
else
    BASE_ARGS="$BASE_ARGS --no-skip-existing"
fi

# Build final command based on mode
if [ "$DEBUG_MODE" = true ]; then
    # Debug mode - single GPU with ipdb
    cmd="python -m ipdb -c continue $SCRIPT_NAME $BASE_ARGS"
    echo "🔧 Debug mode command: $cmd"
elif [ "$TEST_MODE" = true ]; then
    # Test mode - 2 GPUs
    cmd="python $SCRIPT_NAME $BASE_ARGS"
    echo "🧪 Test mode command: $cmd"
else
    # SLURM mode - use all allocated GPUs
    cmd="python $SCRIPT_NAME $BASE_ARGS"
    echo "🚀 SLURM mode command: $cmd"
fi

echo "Command: $cmd"
echo ""

# Run processing
if [ "$DEBUG_MODE" = true ]; then
    echo "===== Running in DEBUG mode ====="
    echo "Using single GPU for debugging"
    echo ""
    
    $cmd
    EXIT_CODE=$?
else
    echo "===== Running HaWoR Batch Processing ====="
    echo "Using $NUM_GPUS GPU(s) for processing"
    echo ""
    
    echo "Command: $cmd"
    echo ""
    
    eval $cmd &
    PYTHON_PID=$!
    echo "Main process started with PID: $PYTHON_PID"
    echo "Use Ctrl+C to stop the processing"
    echo ""
    
    # Wait for the process and capture exit code
    wait $PYTHON_PID
    EXIT_CODE=$?
    
    # Clear the PID variable after process ends
    unset PYTHON_PID
fi

echo ""
echo "===== HaWoR Batch Processing completed ====="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ HaWoR batch processing completed successfully"
    echo ""
    echo "📊 Processing Summary:"
    echo "   Train list: $TRAIN_LIST"
    echo "   Input base: $INPUT_BASE"
    echo "   Output base: $OUTPUT_BASE"
    echo "   GPUs used: $NUM_GPUS ($GPU_LIST)"
    echo "   Max workers per GPU: $MAX_WORKERS_PER_GPU"
    echo "   Skip existing: $SKIP_EXISTING"
    echo ""
    echo "📁 Output directories:"
    echo "   - Hand videos: $OUTPUT_BASE/videos_hands/"
    echo "   - Hand masks: $OUTPUT_BASE/videos_hands_mask/"
    echo "   - Hand overlays: $OUTPUT_BASE/videos_hands_overlay/"
else
    echo "❌ HaWoR batch processing failed with exit code: $EXIT_CODE"
    echo "Check the output above for error details"
    echo ""
    echo "🔍 Debugging tips:"
    echo "   - Check if train_list.txt exists and is readable"
    echo "   - Verify input_base directory exists"
    echo "   - Check GPU availability"
    echo "   - Verify checkpoint and infiller_weight paths"
    echo "   - Try debug mode with --debug"
fi

echo ""
echo "📁 SLURM output files:"
echo "   - Output: out/251105/%j_hawor_process.out"
echo "   - Error: out/251105/%j_hawor_process.err"


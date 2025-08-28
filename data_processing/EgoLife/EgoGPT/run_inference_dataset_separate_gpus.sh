#!/bin/bash

# Change to the EgoGPT directory
cd "$(dirname "$0")"

# Source conda environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate egogpt

# Configuration
PRETRAINED_PATH="lmms-lab/EgoGPT-7b-EgoIT-EgoLife"
DATA_ROOT="./data"
QUERY="Please describe the video in detail."

echo "=== Multi-GPU Separate Inference Runner ==="
echo ""

echo "This script runs inference on separate GPUs with different checklist files."
echo "Checklist files should follow the pattern: video_checklist_partXX.txt"
echo ""

read -p "Enter number of GPUs to use: " num_gpus
read -p "Enter base name for checklist files (default: video_checklist): " checklist_base
checklist_base=${checklist_base:-"video_checklist"}
read -p "Remove processed videos from checklist? (y/n): " remove_choice

# Validate number of GPUs
if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [ "$num_gpus" -lt 1 ]; then
    echo "Error: Number of GPUs must be a positive integer"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Number of GPUs: $num_gpus"
echo "  Checklist base name: $checklist_base"
echo "  Remove from checklist: $remove_choice"
echo ""

# Check if all checklist files exist
echo "Checking checklist files..."
for i in $(seq 0 $((num_gpus-1))); do
    checklist_file="${checklist_base}_part$(printf "%02d" $i).txt"
    if [ ! -f "$checklist_file" ]; then
        echo "Error: Checklist file not found: $checklist_file"
        exit 1
    fi
    echo "  ✅ Found: $checklist_file"
done

echo ""
echo "Starting inference on $num_gpus GPUs..."
echo ""

# Initialize arrays to store PIDs
declare -a PIDS=()

# Start processes for each GPU
for i in $(seq 0 $((num_gpus-1))); do
    checklist_file="${checklist_base}_part$(printf "%02d" $i).txt"
    
    echo "Starting GPU $i process..."
    echo "  Checklist file: $checklist_file"
    
    # Build base command
    CMD="CUDA_VISIBLE_DEVICES=$i python -u inference_dataset.py \
        --data_root $DATA_ROOT \
        --query \"$QUERY\" \
        --pretrained_path $PRETRAINED_PATH \
        --skip_existing \
        --checklist_file \"$checklist_file\" \
        --num_gpus 1"
    
    # Add remove flag if requested
    if [ "$remove_choice" = "y" ] || [ "$remove_choice" = "Y" ]; then
        CMD="$CMD --remove_from_checklist"
    fi
    
    echo "  Command: $CMD"
    echo ""
    
    # Start process in background with separate stdout and stderr logging
    eval $CMD > "gpu${i}_stdout.log" 2> "gpu${i}_stderr.log" &
    PID=$!
    PIDS[$i]=$PID
    
    echo "  GPU $i process started with PID: $PID"
    echo "  Log files: gpu${i}_stdout.log, gpu${i}_stderr.log"
    echo ""
done

echo "All $num_gpus processes are running in background."
echo ""
echo "Monitor progress with:"
for i in $(seq 0 $((num_gpus-1))); do
    echo "  tail -f gpu${i}_stdout.log"
done
echo ""
echo "Monitor errors with:"
for i in $(seq 0 $((num_gpus-1))); do
    echo "  tail -f gpu${i}_stderr.log"
done
echo ""
echo "Check process status with:"
echo "  ps aux | grep inference_dataset"
echo ""
echo "To stop all processes:"
echo "  kill ${PIDS[*]}"
echo ""

# Wait for all processes to complete
echo "Waiting for all processes to complete..."
for i in $(seq 0 $((num_gpus-1))); do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✅ GPU $i completed successfully"
    else
        echo "  ❌ GPU $i failed with exit code $EXIT_CODE"
    fi
done

echo ""
echo "All processes completed!"
echo "Check log files for details:"
for i in $(seq 0 $((num_gpus-1))); do
    echo "  gpu${i}_stdout.log (stdout)"
    echo "  gpu${i}_stderr.log (stderr)"
done
